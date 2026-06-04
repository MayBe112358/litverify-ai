"""DOIDB / China DOI resolver fallback."""
from __future__ import annotations

from services._client_base import BaseCitationClient
from services.rule_engine import Citation
from utils.doi_utils import normalize_doi


class DOIDBClient(BaseCitationClient):
    """Resolve a DOI to a URL when metadata-rich DOI registries miss."""

    BASE = "https://doidb.wdc-terra.org/mds/doi"
    CACHE_NAME = "doidb"
    RETRY_ATTEMPTS = 2
    RETRY_WAIT_MAX = 5

    def by_doi(self, doi: str | None) -> Citation | None:
        doi = normalize_doi(doi)
        if not doi:
            return None
        return self._resolve_doi(doi, self._fetch)

    def _fetch(self, doi: str) -> Citation | None:
        url = self._get_text(f"{self.BASE}/{doi}").strip() or None
        return Citation(doi=doi, url=url, source="DOIDB") if url else None
