"""Wanfang literature search client.

The public documentation requires AppKey/AppSecret request signing. This
client stays disabled until credentials are provided in environment settings.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
from typing import Any

from config.settings import settings
from services._client_base import BaseCitationClient, best_title_match, to_int
from services.api_errors import LookupUnavailable
from services.rule_engine import Citation
from utils.doi_utils import normalize_doi


class WanfangClient(BaseCitationClient):
    """Client for Wanfang's signed paper query endpoint."""

    BASE = "https://api.wanfangdata.com.cn/reader/papers"
    CACHE_NAME = "wanfang"
    RETRY_ATTEMPTS = 2
    RETRY_WAIT_MAX = 5

    @property
    def configured(self) -> bool:
        return bool(settings.wanfang_app_key and settings.wanfang_app_secret)

    def by_doi(self, doi: str | None) -> Citation | None:
        doi = normalize_doi(doi)
        if not doi or not self.configured:
            return None
        return self._lookup(f"doi::{doi}", lambda: self._search_keyword(doi, expected_doi=doi))

    def search_by_title(self, title: str | None, year: int | None = None) -> Citation | None:
        if not title or not self.configured:
            return None
        return self._lookup(
            f"title::{title.lower()}::{year or ''}",
            lambda: self._search_keyword(title, title=title, year=year),
        )

    def _search_keyword(
        self,
        keyword: str,
        title: str | None = None,
        year: int | None = None,
        expected_doi: str | None = None,
    ) -> Citation | None:
        payload = {"keyword": keyword, "page": 1, "type": settings.wanfang_query_type}
        data = self._query(payload)
        if str(data.get("Code", "")).lower() not in {"success", "成功"}:
            raise LookupUnavailable(data.get("Msg") or "Wanfang lookup failed")
        citations = [_to_citation(item.get("periodical") or {}) for item in _page_items(data)]
        if expected_doi:
            for cand in citations:
                if cand.doi and normalize_doi(cand.doi) == expected_doi:
                    return cand
            return None
        if year:
            citations = [c for c in citations if not c.year or c.year == year]
        return best_title_match(title or keyword, citations)

    def _query(self, payload: dict[str, Any]) -> dict[str, Any]:
        body = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        return self._post_json(
            self.BASE, data=body.encode("utf-8"), headers=self._headers(body)
        )

    def _headers(self, body: str) -> dict[str, str]:
        return {
            "Content-Type": "application/json",
            "X-Ca-Version": settings.wanfang_api_version,
            "X-Ca-AppKey": settings.wanfang_app_key,
            "X-Ca-Signature": self._signature(body),
        }

    @staticmethod
    def _signature(body: str) -> str:
        digest = hmac.new(
            settings.wanfang_app_secret.encode("utf-8"),
            body.encode("utf-8"),
            hashlib.sha256,
        ).digest()
        return base64.b64encode(digest).decode("ascii")


def _page_items(data: dict[str, Any]) -> list[dict[str, Any]]:
    page = data.get("pageInfo") or {}
    items = page.get("pageDatas") or []
    if isinstance(items, dict):
        return [items]
    return list(items)


def _to_citation(item: dict[str, Any]) -> Citation:
    return Citation(
        title=item.get("title"),
        authors=[str(a) for a in item.get("creators") or []],
        year=to_int(item.get("publishYear")),
        venue=item.get("periodicalTitle"),
        volume=item.get("volume") or None,
        issue=item.get("issue") or None,
        pages=item.get("page") or item.get("pageNo") or None,
        doi=normalize_doi(item.get("doi")),
        type=item.get("type") or "periodical",
        citation_count=to_int(item.get("citedCount")),
        source="Wanfang",
    )
