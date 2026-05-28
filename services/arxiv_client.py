"""arXiv API client for preprint lookup."""
from __future__ import annotations

import xml.etree.ElementTree as ET
from typing import Any

import requests
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from config.settings import settings
from services.api_errors import LookupUnavailable, is_transient_error
from services.rule_engine import Citation
from utils.cache import make_disk_cache


CACHE = make_disk_cache(".cache/arxiv")
NS = {"atom": "http://www.w3.org/2005/Atom"}


class ArxivClient:
    """Minimal arXiv Atom API wrapper."""

    BASE = "https://export.arxiv.org/api/query"

    def __init__(self) -> None:
        self.session = requests.Session()
        # arXiv 的礼貌使用条款要求带可识别 User-Agent。
        mail = settings.crossref_email or "litverify@example.com"
        self.session.headers.update(
            {"User-Agent": f"LitVerify-AI/1.0 (mailto:{mail})"}
        )

    def by_id(self, arxiv_id: str | None) -> Citation | None:
        """Resolve an arXiv id (None = miss, raise = unavailable)."""
        if not arxiv_id:
            return None
        key = f"id::{arxiv_id.lower()}"
        if key in CACHE:
            return CACHE[key]  # type: ignore[return-value]
        citation = self._first_entry(self._fetch({"id_list": arxiv_id}))
        CACHE.set(key, citation, expire=86400)
        return citation

    def search_by_title(self, title: str | None) -> Citation | None:
        """Search arXiv by title."""
        if not title:
            return None
        key = f"title::{title.lower()}"
        if key in CACHE:
            return CACHE[key]  # type: ignore[return-value]
        citation = self._first_entry(self._fetch({"search_query": f'ti:"{title}"', "max_results": 3}))
        CACHE.set(key, citation, expire=86400)
        return citation

    def _fetch(self, params: dict[str, Any]) -> str:
        try:
            return self._get(params)
        except requests.RequestException as exc:
            raise LookupUnavailable(str(exc)) from exc

    @retry(
        retry=retry_if_exception(is_transient_error),
        stop=stop_after_attempt(2),
        wait=wait_exponential(min=1, max=5),
    )
    def _get(self, params: dict[str, Any]) -> str:
        response = self.session.get(self.BASE, params=params, timeout=settings.api_timeout)
        response.raise_for_status()
        return response.text

    @staticmethod
    def _first_entry(xml_text: str) -> Citation | None:
        root = ET.fromstring(xml_text)
        entry = root.find("atom:entry", NS)
        if entry is None:
            return None
        title = (entry.findtext("atom:title", default="", namespaces=NS) or "").strip()
        published = entry.findtext("atom:published", default="", namespaces=NS) or ""
        authors = [
            node.findtext("atom:name", default="", namespaces=NS)
            for node in entry.findall("atom:author", NS)
        ]
        link = entry.find("atom:id", NS)
        arxiv_id = (link.text.rsplit("/", 1)[-1] if link is not None and link.text else None)
        return Citation(
            title=" ".join(title.split()),
            authors=[a for a in authors if a],
            year=int(published[:4]) if published[:4].isdigit() else None,
            venue="arXiv",
            arxiv_id=arxiv_id,
            url=link.text if link is not None else None,
            type="preprint",
            source="arXiv",
        )
