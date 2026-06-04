"""OpenAlex API client used as a CrossRef companion source."""
from __future__ import annotations

from typing import Any

from config.settings import settings
from services._client_base import BaseCitationClient
from services.rule_engine import Citation
from utils.doi_utils import normalize_doi


class OpenAlexClient(BaseCitationClient):
    """Small wrapper around OpenAlex works endpoints."""

    BASE = "https://api.openalex.org"
    CACHE_NAME = "openalex"

    def by_doi(self, doi: str | None) -> Citation | None:
        """Resolve DOI through OpenAlex (None = miss, raise = unavailable)."""
        doi = normalize_doi(doi)
        if not doi:
            return None
        return self._resolve_doi(
            doi,
            lambda d: self._to_citation(self._get_json(f"{self.BASE}/works/https://doi.org/{d}")),
        )

    def search_by_title(self, title: str | None, year: int | None = None) -> Citation | None:
        """Search OpenAlex by title with title-similarity post-filter."""
        if not title:
            return None

        def fetch() -> list[Citation]:
            params: dict[str, Any] = {"search": title, "per-page": 10}
            if settings.openalex_email:
                params["mailto"] = settings.openalex_email
            if year:
                params["filter"] = f"publication_year:{year}"
            data = self._get_json(f"{self.BASE}/works", params=params)
            return [self._to_citation(item) for item in data.get("results") or []]

        return self._search_titles(title, year, fetch)

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
