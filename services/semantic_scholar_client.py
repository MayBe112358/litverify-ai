"""Semantic Scholar Academic Graph API client."""
from __future__ import annotations

from typing import Any

from config.settings import settings
from services._client_base import BaseCitationClient
from services.rule_engine import Citation
from utils.doi_utils import normalize_doi


FIELDS = "title,authors,year,venue,externalIds,citationCount,publicationTypes,journal,url"


class SemanticScholarClient(BaseCitationClient):
    """Resolve or search papers through Semantic Scholar."""

    BASE = "https://api.semanticscholar.org/graph/v1"
    CACHE_NAME = "semantic_scholar"

    def _configure_session(self) -> None:
        if settings.semantic_scholar_api_key:
            self.session.headers["x-api-key"] = settings.semantic_scholar_api_key

    def by_doi(self, doi: str | None) -> Citation | None:
        doi = normalize_doi(doi)
        if not doi:
            return None
        return self._resolve_doi(
            doi,
            lambda d: self._to_citation(
                self._get_json(f"{self.BASE}/paper/DOI:{d}", params={"fields": FIELDS})
            ),
        )

    def search_by_title(self, title: str | None, year: int | None = None) -> Citation | None:
        if not title:
            return None

        def fetch() -> list[Citation]:
            params: dict[str, Any] = {"query": title, "limit": 10, "fields": FIELDS}
            if year:
                params["year"] = str(year)
            data = self._get_json(f"{self.BASE}/paper/search", params=params)
            return [self._to_citation(item) for item in data.get("data") or []]

        return self._search_titles(title, year, fetch)

    @staticmethod
    def _to_citation(item: dict[str, Any]) -> Citation:
        external = item.get("externalIds") or {}
        journal = item.get("journal") or {}
        return Citation(
            title=item.get("title"),
            authors=[a.get("name", "") for a in item.get("authors", []) if a.get("name")],
            year=item.get("year"),
            venue=journal.get("name") or item.get("venue"),
            doi=normalize_doi(external.get("DOI")),
            url=item.get("url"),
            type=", ".join(item.get("publicationTypes") or []) or "article",
            citation_count=item.get("citationCount"),
            source="Semantic Scholar",
        )
