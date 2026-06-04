"""DataCite DOI metadata client."""
from __future__ import annotations

from typing import Any

from services._client_base import BaseCitationClient, to_int
from services.rule_engine import Citation
from utils.doi_utils import normalize_doi


class DataCiteClient(BaseCitationClient):
    """Resolve and search DataCite DOI metadata."""

    BASE = "https://api.datacite.org"
    CACHE_NAME = "datacite"

    def by_doi(self, doi: str | None) -> Citation | None:
        doi = normalize_doi(doi)
        if not doi:
            return None
        return self._resolve_doi(
            doi,
            lambda d: self._to_citation(
                (self._get_json(f"{self.BASE}/dois/{d}").get("data") or {}).get("attributes") or {}
            ),
        )

    def search_by_title(self, title: str | None, year: int | None = None) -> Citation | None:
        if not title:
            return None

        def fetch() -> list[Citation]:
            params: dict[str, Any] = {"query": title, "page[size]": 10}
            if year:
                params["query"] = f"{title} {year}"
            data = self._get_json(f"{self.BASE}/dois", params=params)
            return [
                self._to_citation(item.get("attributes") or {})
                for item in (data.get("data") or [])
            ]

        return self._search_titles(title, year, fetch)

    @staticmethod
    def _to_citation(item: dict[str, Any]) -> Citation:
        types = item.get("types") or {}
        return Citation(
            title=_first_title(item.get("titles")),
            authors=_creators(item.get("creators")),
            year=to_int(item.get("publicationYear")),
            venue=item.get("publisher"),
            doi=normalize_doi(item.get("doi")),
            url=item.get("url"),
            type=types.get("resourceTypeGeneral") or types.get("resourceType") or "unknown",
            source="DataCite",
        )


def _first_title(value: Any) -> str | None:
    if not value:
        return None
    first = value[0] if isinstance(value, list) else value
    if isinstance(first, dict):
        return first.get("title")
    return str(first)


def _creators(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    names = []
    for creator in value:
        if isinstance(creator, dict):
            name = creator.get("name")
            if not name:
                given = creator.get("givenName")
                family = creator.get("familyName")
                name = " ".join(part for part in (given, family) if part)
            if name:
                names.append(str(name))
    return names
