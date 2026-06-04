"""arXiv API client for preprint lookup."""
from __future__ import annotations

import xml.etree.ElementTree as ET

from config.settings import settings
from services._client_base import BaseCitationClient
from services.rule_engine import Citation


NS = {"atom": "http://www.w3.org/2005/Atom"}


class ArxivClient(BaseCitationClient):
    """Minimal arXiv Atom API wrapper."""

    BASE = "https://export.arxiv.org/api/query"
    CACHE_NAME = "arxiv"
    RETRY_ATTEMPTS = 2
    RETRY_WAIT_MAX = 5

    def _configure_session(self) -> None:
        # arXiv 的礼貌使用条款要求带可识别 User-Agent。
        mail = settings.crossref_email or "litverify@example.com"
        self.session.headers.update({"User-Agent": f"LitVerify-AI/1.0 (mailto:{mail})"})

    def by_id(self, arxiv_id: str | None) -> Citation | None:
        """Resolve an arXiv id (None = miss, raise = unavailable)."""
        if not arxiv_id:
            return None
        return self._lookup(
            f"id::{arxiv_id.lower()}",
            lambda: self._first_entry(self._get_text(self.BASE, {"id_list": arxiv_id})),
        )

    def search_by_title(self, title: str | None) -> Citation | None:
        """Search arXiv by title."""
        if not title:
            return None
        return self._lookup(
            f"title::{title.lower()}",
            lambda: self._first_entry(
                self._get_text(self.BASE, {"search_query": f'ti:"{title}"', "max_results": 3})
            ),
        )

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
