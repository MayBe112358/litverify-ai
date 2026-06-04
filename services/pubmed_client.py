"""PubMed / NCBI E-utilities client for biomedical citation lookup."""
from __future__ import annotations

import re
from typing import Any

from config.settings import settings
from services._client_base import BaseCitationClient
from services.rule_engine import Citation
from utils.doi_utils import normalize_doi


class PubMedClient(BaseCitationClient):
    """Small wrapper around NCBI ESearch + ESummary."""

    BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
    CACHE_NAME = "pubmed"

    def _configure_session(self) -> None:
        self.session.headers.update({"User-Agent": "LitVerify-AI/1.0"})

    def by_doi(self, doi: str | None) -> Citation | None:
        doi = normalize_doi(doi)
        if not doi:
            return None

        def fetch() -> Citation | None:
            ids = self._search_ids(f"{doi}[doi]", retmax=3)
            return self._summary(ids[0]) if ids else None

        return self._lookup(f"doi::{doi}", fetch)

    def search_by_title(self, title: str | None, year: int | None = None) -> Citation | None:
        if not title:
            return None

        def fetch() -> list[Citation]:
            term = f'"{title}"[Title]'
            if year:
                term += f" AND {year}[pdat]"
            ids = self._search_ids(term, retmax=8)
            return [self._summary(uid) for uid in ids]

        return self._search_titles(title, year, fetch)

    def _search_ids(self, term: str, retmax: int = 5) -> list[str]:
        params = {
            "db": "pubmed",
            "term": term,
            "retmode": "json",
            "retmax": retmax,
            "sort": "relevance",
        }
        if settings.ncbi_api_key:
            params["api_key"] = settings.ncbi_api_key
        data = self._get_json(f"{self.BASE}/esearch.fcgi", params=params)
        return list((data.get("esearchresult") or {}).get("idlist") or [])

    def _summary(self, uid: str) -> Citation:
        params = {"db": "pubmed", "id": uid, "retmode": "json"}
        if settings.ncbi_api_key:
            params["api_key"] = settings.ncbi_api_key
        data = self._get_json(f"{self.BASE}/esummary.fcgi", params=params)
        item = (data.get("result") or {}).get(uid) or {}
        return self._to_citation(item, uid)

    @staticmethod
    def _to_citation(item: dict[str, Any], uid: str | None = None) -> Citation:
        article_ids = item.get("articleids") or []
        doi = None
        for ident in article_ids:
            if (ident.get("idtype") or "").lower() == "doi":
                doi = normalize_doi(ident.get("value"))
                break
        authors = [a.get("name", "") for a in item.get("authors", [])]
        return Citation(
            title=_clean_title(item.get("title")),
            authors=[a for a in authors if a],
            year=_year(item.get("pubdate") or item.get("epubdate")),
            venue=item.get("fulljournalname") or item.get("source"),
            volume=item.get("volume") or None,
            issue=item.get("issue") or None,
            pages=item.get("pages") or None,
            doi=doi,
            url=f"https://pubmed.ncbi.nlm.nih.gov/{uid}/" if uid else None,
            type=", ".join(item.get("pubtype") or []) or "article",
            source="PubMed",
        )


def _clean_title(value: str | None) -> str | None:
    if not value:
        return None
    return re.sub(r"\s+", " ", value).strip().rstrip(".")


def _year(value: str | None) -> int | None:
    if not value:
        return None
    match = re.search(r"(18|19|20|21)\d{2}", value)
    return int(match.group(0)) if match else None
