"""DBLP publication search client for computer-science citations."""
from __future__ import annotations

from typing import Any

from services._client_base import BaseCitationClient, to_int
from services.rule_engine import Citation
from utils.doi_utils import normalize_doi


class DBLPClient(BaseCitationClient):
    """Wrapper around DBLP's public publication search API."""

    BASE = "https://dblp.org/search/publ/api"
    CACHE_NAME = "dblp"

    def by_doi(self, doi: str | None) -> Citation | None:
        doi = normalize_doi(doi)
        if not doi:
            return None
        return self._lookup(f"doi::{doi}", lambda: self._search_exact_doi(doi))

    def search_by_title(self, title: str | None, year: int | None = None) -> Citation | None:
        if not title:
            return None

        def fetch() -> list[Citation]:
            data = self._get_json(self.BASE, params={"q": title, "format": "json", "h": 10})
            candidates = [_to_citation(hit.get("info") or {}) for hit in _hits(data)]
            if year:
                candidates = [c for c in candidates if not c.year or c.year == year]
            return candidates

        return self._search_titles(title, year, fetch)

    def _search_exact_doi(self, doi: str) -> Citation | None:
        data = self._get_json(self.BASE, params={"q": doi, "format": "json", "h": 10})
        for hit in _hits(data):
            citation = _to_citation(hit.get("info") or {})
            if citation.doi and normalize_doi(citation.doi) == doi:
                return citation
        return None


def _hits(data: dict[str, Any]) -> list[dict[str, Any]]:
    hits = (((data.get("result") or {}).get("hits") or {}).get("hit") or [])
    if isinstance(hits, dict):
        return [hits]
    return list(hits)


def _to_citation(item: dict[str, Any]) -> Citation:
    return Citation(
        title=_text(item.get("title")).rstrip(".") or None,
        authors=_authors(item.get("authors")),
        year=to_int(item.get("year")),
        venue=_text(item.get("venue")) or None,
        pages=_text(item.get("pages")) or None,
        doi=normalize_doi(_text(item.get("doi"))),
        url=_text(item.get("url")) or None,
        type=_text(item.get("type")) or "article",
        source="DBLP",
    )


def _text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, dict):
        return str(value.get("text") or "")
    return str(value)


def _authors(value: Any) -> list[str]:
    if not value:
        return []
    authors = value.get("author") if isinstance(value, dict) else value
    if isinstance(authors, dict):
        authors = [authors]
    if not isinstance(authors, list):
        authors = [authors]
    return [name for item in authors if (name := _text(item).strip())]
