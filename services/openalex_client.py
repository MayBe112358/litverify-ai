"""OpenAlex API client used as a CrossRef companion source."""
from __future__ import annotations

from typing import Any

import requests
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from config.settings import settings
from services.api_errors import LookupUnavailable, is_transient_error
from services.rule_engine import Citation
from utils.cache import make_disk_cache
from utils.doi_utils import normalize_doi
from utils.text_similarity import title_similarity


CACHE = make_disk_cache(".cache/openalex")
TITLE_SIMILARITY_FLOOR = 0.70


class OpenAlexClient:
    """Small wrapper around OpenAlex works endpoints."""

    BASE = "https://api.openalex.org"

    def __init__(self) -> None:
        self.session = requests.Session()

    def by_doi(self, doi: str | None) -> Citation | None:
        """Resolve DOI through OpenAlex (None = miss, raise = unavailable)."""
        doi = normalize_doi(doi)
        if not doi:
            return None
        key = f"doi::{doi}"
        if key in CACHE:
            return CACHE[key]  # type: ignore[return-value]
        try:
            data = self._get(f"{self.BASE}/works/https://doi.org/{doi}")
        except requests.HTTPError as exc:
            if exc.response is not None and exc.response.status_code < 500:
                CACHE.set(key, None, expire=86400)  # confirmed miss
                return None
            raise LookupUnavailable(str(exc)) from exc
        except requests.RequestException as exc:
            raise LookupUnavailable(str(exc)) from exc
        citation = self._to_citation(data)
        CACHE.set(key, citation, expire=86400)
        return citation

    def search_by_title(self, title: str | None, year: int | None = None) -> Citation | None:
        """Search OpenAlex by title with title-similarity post-filter."""
        if not title:
            return None
        key = f"title::{title.lower()}::{year or ''}"
        if key in CACHE:
            return CACHE[key]  # type: ignore[return-value]
        params: dict[str, Any] = {"search": title, "per-page": 10}
        if settings.openalex_email:
            params["mailto"] = settings.openalex_email
        if year:
            params["filter"] = f"publication_year:{year}"
        try:
            data = self._get(f"{self.BASE}/works", params=params)
        except requests.RequestException as exc:
            raise LookupUnavailable(str(exc)) from exc
        results = data.get("results") or []
        best: Citation | None = None
        best_score = 0.0
        for item in results:
            cand = self._to_citation(item)
            score = title_similarity(title, cand.title)
            if score > best_score:
                best = cand
                best_score = score
        citation = best if best_score >= TITLE_SIMILARITY_FLOOR else None
        CACHE.set(key, citation, expire=86400)
        return citation

    @retry(
        retry=retry_if_exception(is_transient_error),
        stop=stop_after_attempt(3),
        wait=wait_exponential(min=1, max=8),
    )
    def _get(self, url: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        response = self.session.get(url, params=params, timeout=settings.api_timeout)
        response.raise_for_status()
        return response.json()

    @staticmethod
    def _to_citation(item: dict[str, Any]) -> Citation:
        authors = [a.get("author", {}).get("display_name", "") for a in item.get("authorships", [])]
        biblio = item.get("biblio") or {}
        primary_location = item.get("primary_location") or {}
        source = primary_location.get("source") or {}
        host_venue = item.get("host_venue") or {}
        venue = source.get("display_name") or host_venue.get("display_name")
        first_page = biblio.get("first_page")
        last_page = biblio.get("last_page")
        return Citation(
            title=item.get("title"),
            authors=[a for a in authors if a],
            year=item.get("publication_year"),
            venue=venue,
            volume=biblio.get("volume"),
            issue=biblio.get("issue"),
            pages=f"{first_page}-{last_page}" if first_page and last_page else first_page,
            doi=normalize_doi(item.get("doi")),
            type=item.get("type") or "article",
            citation_count=item.get("cited_by_count"),
            source="OpenAlex",
        )
