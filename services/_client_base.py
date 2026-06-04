"""Shared plumbing for the external citation-source clients.

Every source client (CrossRef, OpenAlex, PubMed, ...) repeats the same three
chores: a retrying HTTP GET/POST, a 24h disk cache around each lookup, and the
"404 means a real miss / timeout means unavailable" branching that keeps an
outage from being mistaken for "this DOI doesn't exist". This module owns that
plumbing once so each client only has to express the two things that are
genuinely source-specific: how to build the request, and how to parse a hit
into a :class:`Citation`.
"""
from __future__ import annotations

from typing import Any, Callable

import requests
from tenacity import Retrying, retry_if_exception, stop_after_attempt, wait_exponential

from config.settings import settings
from services.api_errors import LookupUnavailable, is_transient_error
from services.rule_engine import Citation
from utils.cache import make_disk_cache
from utils.text_similarity import title_similarity


# A search hit's top result isn't necessarily the same paper (relevance ranking
# is loose), so we re-score candidates by title similarity and drop anything
# below this floor. 0.70 tolerates case/spacing/punctuation drift while still
# rejecting same-topic-but-different-paper near misses.
TITLE_SIMILARITY_FLOOR = 0.70
CACHE_TTL = 86400  # 24h — long enough to be useful, short enough to self-heal.


def to_int(value: Any) -> int | None:
    """Best-effort int coercion that swallows the usual junk (None, ""...)."""
    try:
        return int(value)
    except Exception:
        return None


def best_title_match(title: str, candidates: list[Citation]) -> Citation | None:
    """Pick the candidate whose title is most similar, or None below the floor."""
    best: Citation | None = None
    best_score = 0.0
    for cand in candidates:
        score = title_similarity(title, cand.title)
        if score > best_score:
            best = cand
            best_score = score
    return best if best_score >= TITLE_SIMILARITY_FLOOR else None


class BaseCitationClient:
    """Base class wrapping cache + retrying HTTP + miss/unavailable semantics.

    Subclasses set :attr:`CACHE_NAME` (and optionally tune the retry budget),
    then implement their request building / parsing on top of :meth:`_cached`,
    :meth:`_get_json` / :meth:`_get_text`, :meth:`_resolve_doi` and
    :meth:`_search_titles`.
    """

    CACHE_NAME: str = ""
    RETRY_ATTEMPTS: int = 3
    RETRY_WAIT_MAX: int = 8

    def __init__(self) -> None:
        if not self.CACHE_NAME:
            raise ValueError(f"{type(self).__name__} must set CACHE_NAME")
        self.cache = make_disk_cache(f".cache/{self.CACHE_NAME}")
        self.session = requests.Session()
        self._retryer = Retrying(
            retry=retry_if_exception(is_transient_error),
            stop=stop_after_attempt(self.RETRY_ATTEMPTS),
            wait=wait_exponential(min=1, max=self.RETRY_WAIT_MAX),
            # Re-raise the *original* exception (e.g. requests.ConnectionError)
            # once retries are exhausted, instead of tenacity's RetryError
            # wrapper. The lookup templates below catch requests.RequestException
            # to map outages → LookupUnavailable; a RetryError would slip past
            # them and crash the whole verification on a single source outage.
            reraise=True,
        )
        self._configure_session()

    def _configure_session(self) -> None:
        """Hook for subclasses to add headers / auth. No-op by default."""

    # -- caching -----------------------------------------------------------
    def _cached(self, key: str, produce: Callable[[], Citation | None]) -> Citation | None:
        """Return a cached value or compute, store (incl. ``None`` misses) and return.

        ``produce`` may raise :class:`LookupUnavailable`; that propagates
        without being cached, so a transient outage never poisons the cache.
        """
        if key in self.cache:
            return self.cache[key]  # type: ignore[return-value]
        value = produce()
        self.cache.set(key, value, expire=CACHE_TTL)
        return value

    # -- HTTP --------------------------------------------------------------
    def _request(
        self, method: str, url: str, params: dict[str, Any] | None = None, **kwargs: Any
    ) -> requests.Response:
        response = self.session.request(
            method, url, params=params, timeout=settings.api_timeout, **kwargs
        )
        response.raise_for_status()
        return response

    def _get_response(self, url: str, params: dict[str, Any] | None = None) -> requests.Response:
        return self._retryer(self._request, "GET", url, params)

    def _get_json(self, url: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        return self._get_response(url, params).json()

    def _get_text(self, url: str, params: dict[str, Any] | None = None) -> str:
        return self._get_response(url, params).text

    def _post_json(self, url: str, *, data: Any, headers: dict[str, str]) -> dict[str, Any]:
        return self._retryer(self._request, "POST", url, None, data=data, headers=headers).json()

    # -- lookup templates --------------------------------------------------
    def _lookup(self, key: str, fetch: Callable[[], Citation | None]) -> Citation | None:
        """Cache ``fetch()`` under ``key``, mapping network errors to
        :class:`LookupUnavailable` so transient outages aren't cached as misses.
        """
        def produce() -> Citation | None:
            try:
                return fetch()
            except requests.RequestException as exc:
                raise LookupUnavailable(str(exc)) from exc

        return self._cached(key, produce)

    def _resolve_doi(
        self, doi: str, build: Callable[[str], Citation | None]
    ) -> Citation | None:
        """Cache a DOI lookup, treating a 4xx as a real miss and 5xx/network
        failures as :class:`LookupUnavailable` (not evidence of absence).

        ``build(doi)`` performs the HTTP call + parse and may raise
        ``requests.HTTPError`` / ``requests.RequestException``.
        """
        def produce() -> Citation | None:
            try:
                return build(doi)
            except requests.HTTPError as exc:
                if exc.response is not None and exc.response.status_code < 500:
                    return None  # confirmed miss
                raise LookupUnavailable(str(exc)) from exc
            except requests.RequestException as exc:
                raise LookupUnavailable(str(exc)) from exc

        return self._cached(f"doi::{doi}", produce)

    def _search_titles(
        self,
        title: str,
        year: int | None,
        fetch: Callable[[], list[Citation]],
    ) -> Citation | None:
        """Cache a title search: fetch candidates, then keep the best match.

        ``fetch()`` returns parsed candidates and may raise
        ``requests.RequestException`` (turned into :class:`LookupUnavailable`).
        """
        return self._lookup(
            f"title::{title.lower()}::{year or ''}",
            lambda: best_title_match(title, fetch()),
        )
