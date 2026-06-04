"""CrossRef API client with polite headers, retry and 24h cache."""
from __future__ import annotations

from typing import Any

from config.settings import settings
from services._client_base import BaseCitationClient
from services.rule_engine import Citation
from utils.doi_utils import normalize_doi


class CrossRefClient(BaseCitationClient):
    """Small wrapper around CrossRef works endpoints."""

    BASE = "https://api.crossref.org"
    CACHE_NAME = "crossref"

    def _configure_session(self) -> None:
        mail = settings.crossref_email or "litverify@example.com"
        self.session.headers.update({"User-Agent": f"LitVerify-AI/1.0 (mailto:{mail})"})

    def by_doi(self, doi: str | None) -> Citation | None:
        """Resolve one DOI to a Citation (None = confirmed miss)."""
        doi = normalize_doi(doi)
        if not doi:
            return None
        return self._resolve_doi(
            doi,
            lambda d: self._to_citation(self._get_json(f"{self.BASE}/works/{d}").get("message") or {}),
        )

    def search_by_title(self, title: str | None, year: int | None = None) -> Citation | None:
        """Search CrossRef by bibliographic title and optional year.

        ``query.bibliographic`` gives stricter relevance scoring than
        ``query.title``, ``select`` trims the payload, and the shared
        title-similarity floor drops loose near-misses.
        """
        if not title:
            return None

        def fetch() -> list[Citation]:
            params: dict[str, Any] = {
                "query.bibliographic": title,
                "rows": 10,
                "select": "DOI,title,author,container-title,issued,published-print,"
                          "published-online,volume,issue,page,type,is-referenced-by-count",
            }
            if year:
                params["filter"] = f"from-pub-date:{year},until-pub-date:{year}"
            data = self._get_json(f"{self.BASE}/works", params=params)
            items = (data.get("message") or {}).get("items") or []
            return [self._to_citation(item) for item in items]

        return self._search_titles(title, year, fetch)

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
