"""CrossRef API client with polite headers, retry and 24h cache."""
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


CACHE = make_disk_cache(".cache/crossref")
# 搜索结果命中第一名也未必是同一篇论文（CrossRef 的相关性排序很宽松）。
# 低于该阈值的候选直接判为"未命中"，避免把无关文献当作权威记录参与评分。
# 0.70 是经验值：刚好能容下大小写、空格、标点不一致，但拒绝"Deep Quantum
# Learning" 这种主题词同向但实际另一篇论文的弱匹配。
TITLE_SIMILARITY_FLOOR = 0.70


class CrossRefClient:
    """Small wrapper around CrossRef works endpoints."""

    BASE = "https://api.crossref.org"

    def __init__(self) -> None:
        self.session = requests.Session()
        mail = settings.crossref_email or "litverify@example.com"
        self.session.headers.update(
            {"User-Agent": f"LitVerify-AI/1.0 (mailto:{mail})"}
        )

    def by_doi(self, doi: str | None) -> Citation | None:
        """Resolve one DOI to a Citation.

        Returns ``None`` (and caches it) for a genuine 404 miss; raises
        :class:`LookupUnavailable` on transient failures so the caller can
        stay neutral instead of treating an outage as "DOI doesn't exist".
        """
        doi = normalize_doi(doi)
        if not doi:
            return None
        key = f"doi::{doi}"
        if key in CACHE:
            return CACHE[key]  # type: ignore[return-value]
        try:
            data = self._get(f"{self.BASE}/works/{doi}")
        except requests.HTTPError as exc:
            if exc.response is not None and exc.response.status_code < 500:
                CACHE.set(key, None, expire=86400)  # confirmed miss
                return None
            raise LookupUnavailable(str(exc)) from exc
        except requests.RequestException as exc:
            raise LookupUnavailable(str(exc)) from exc
        citation = self._to_citation(data.get("message") or {})
        CACHE.set(key, citation, expire=86400)
        return citation

    def search_by_title(self, title: str | None, year: int | None = None) -> Citation | None:
        """Search CrossRef by bibliographic title and optional year.

        优化点：
        - 用 ``query.bibliographic``（而不是 ``query.title``）让 CrossRef 用更严格的
          全文献相关性算分。
        - ``select`` 限制返回字段，加速 + 礼貌。
        - 拿到候选后，按 ``title_similarity`` 重排序，低于阈值的直接当未命中，
          避免出现"Attention is all you need" 误匹配到 "All you need is love?" 这种情况。
        """
        if not title:
            return None
        key = f"title::{title.lower()}::{year or ''}"
        if key in CACHE:
            return CACHE[key]  # type: ignore[return-value]
        params: dict[str, Any] = {
            "query.bibliographic": title,
            "rows": 10,
            "select": "DOI,title,author,container-title,issued,published-print,"
                      "published-online,volume,issue,page,type,is-referenced-by-count",
        }
        if year:
            params["filter"] = f"from-pub-date:{year},until-pub-date:{year}"
        try:
            data = self._get(f"{self.BASE}/works", params=params)
        except requests.RequestException as exc:
            raise LookupUnavailable(str(exc)) from exc
        items = (data.get("message") or {}).get("items") or []
        best: Citation | None = None
        best_score = 0.0
        for item in items:
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
    def _first_date(item: dict[str, Any]) -> int | None:
        for key in ("published-print", "published-online", "issued", "created"):
            parts = ((item.get(key) or {}).get("date-parts") or [[]])[0]
            if parts:
                try:
                    return int(parts[0])
                except Exception:
                    continue
        return None

    @classmethod
    def _to_citation(cls, item: dict[str, Any]) -> Citation:
        authors = [
            " ".join(part for part in [a.get("given"), a.get("family")] if part).strip()
            for a in item.get("author", [])
        ]
        title = (item.get("title") or [None])[0]
        venue = (item.get("container-title") or [None])[0]
        return Citation(
            title=title,
            authors=[a for a in authors if a],
            year=cls._first_date(item),
            venue=venue,
            volume=item.get("volume"),
            issue=item.get("issue"),
            pages=item.get("page"),
            doi=normalize_doi(item.get("DOI")),
            type=item.get("type", "article"),
            citation_count=item.get("is-referenced-by-count"),
            source="CrossRef",
        )
