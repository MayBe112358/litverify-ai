"""Single-citation verification pipeline."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from config.settings import settings
from db.history import save_history
from llm.insight_generator import explain_verification
from services.api_errors import LookupUnavailable
from services.arxiv_client import ArxivClient
from services.citation_parser import CitationParser
from services.crossref_client import CrossRefClient
from services.datacite_client import DataCiteClient
from services.dblp_client import DBLPClient
from services.doidb_client import DOIDBClient
from services.openalex_client import OpenAlexClient
from services.pubmed_client import PubMedClient
from services.rule_engine import (
    Citation,
    RuleConfig,
    RuleEngine,
    VerificationEvidence,
    VerificationReport,
)
from services.semantic_scholar_client import SemanticScholarClient
from services.wanfang_client import WanfangClient


# Module-level executor reused across every CitationVerifier instance and
# every batch row — avoids the overhead of creating/destroying 3 threads
# per evidence collection (200 rows × 3 threads previously meant 600
# thread spin-ups). The expanded source set can issue up to 9 lookups per
# citation, so a larger shared pool keeps batch validation from serializing.
_EVIDENCE_EXECUTOR = ThreadPoolExecutor(max_workers=24, thread_name_prefix="lv-evidence")


@dataclass(frozen=True)
class SourceSpec:
    """How one external source should be queried."""

    name: str
    client_attr: str
    doi_lookup: bool = True
    title_lookup: bool = True
    fallback_after_doi_miss: bool = False
    arxiv_id_lookup: bool = False
    title_accepts_year: bool = True
    skip_title_when_doi: bool = False


_SOURCE_SPECS = (
    SourceSpec("crossref", "crossref"),
    SourceSpec("openalex", "openalex"),
    SourceSpec(
        "arxiv",
        "arxiv",
        doi_lookup=False,
        arxiv_id_lookup=True,
        title_accepts_year=False,
        skip_title_when_doi=True,
    ),
    SourceSpec("pubmed", "pubmed", fallback_after_doi_miss=True),
    SourceSpec("semantic_scholar", "semantic_scholar", fallback_after_doi_miss=True),
    SourceSpec("dblp", "dblp", fallback_after_doi_miss=True),
    SourceSpec("wanfang", "wanfang", fallback_after_doi_miss=True),
    SourceSpec("datacite", "datacite"),
    SourceSpec("doidb", "doidb", title_lookup=False),
)


def _session_thresholds() -> dict[str, int] | None:
    defaults = settings.default_thresholds
    try:
        import streamlit as st

        real = int(st.session_state.get("real_threshold", defaults["real"]))
        suspicious = int(st.session_state.get("suspicious_threshold", defaults["suspicious"]))
        # Read-side clamp：永远保证 suspicious < real，避免把不合法的阈值传给
        # 规则引擎。注意：这里不会写回 st.session_state，因为 real/suspicious
        # 是侧边栏滑块的 widget key，写入会触发 "widget 已实例化后不能写入"。
        if suspicious >= real:
            suspicious = max(0, real - 10)
        return {"real": real, "suspicious": suspicious}
    except Exception:
        return None


def _session_rules() -> list[RuleConfig] | None:
    """Read live rule overrides from the configuration page (if any)."""
    try:
        import streamlit as st

        overrides = st.session_state.get("rule_overrides_v2")
        if not overrides:
            return None
        return [
            RuleConfig(
                id=item["id"],
                name=item.get("name", item["id"]),
                enabled=bool(item.get("enabled", True)),
                weight=float(item.get("weight", 1)),
                params=dict(item.get("params") or {}),
            )
            for item in overrides
        ]
    except Exception:
        return None


class CitationVerifier:
    """Parse, fetch evidence, score and explain one citation."""

    def __init__(
        self,
        parser: CitationParser | None = None,
        rule_engine: RuleEngine | None = None,
        crossref: CrossRefClient | None = None,
        openalex: OpenAlexClient | None = None,
        arxiv: ArxivClient | None = None,
        pubmed: PubMedClient | None = None,
        semantic_scholar: SemanticScholarClient | None = None,
        dblp: DBLPClient | None = None,
        wanfang: WanfangClient | None = None,
        datacite: DataCiteClient | None = None,
        doidb: DOIDBClient | None = None,
    ) -> None:
        self.parser = parser or CitationParser()
        self.rule_engine = rule_engine or RuleEngine(
            rules=_session_rules(),
            thresholds=_session_thresholds(),
        )
        self.crossref = crossref or CrossRefClient()
        self.openalex = openalex or OpenAlexClient()
        self.arxiv = arxiv or ArxivClient()
        self.pubmed = pubmed or PubMedClient()
        self.semantic_scholar = semantic_scholar or SemanticScholarClient()
        self.dblp = dblp or DBLPClient()
        self.wanfang = wanfang or WanfangClient()
        self.datacite = datacite or DataCiteClient()
        self.doidb = doidb or DOIDBClient()

    def verify(
        self,
        raw_text: str,
        with_llm_explain: bool = True,
        save: bool = True,
    ) -> VerificationReport:
        """Run the full verification pipeline for one raw citation string."""
        citation = self.parser.parse(raw_text)
        return self.verify_citation(citation, with_llm_explain=with_llm_explain, save=save)

    def verify_citation(
        self,
        citation: Citation,
        with_llm_explain: bool = True,
        save: bool = True,
    ) -> VerificationReport:
        """Verify an already-structured :class:`Citation` — used by the
        structured batch path where each spreadsheet row carries field
        columns (title/author/year/...) instead of a free-text citation.
        """
        evidence = self._collect_evidence(citation)
        report = self.rule_engine.evaluate(citation, evidence)
        if with_llm_explain:
            explanation = explain_verification(report)
            report.explanation = explanation
            report.suggestions = list(explanation.get("repair_suggestions") or [])
        else:
            report.suggestions = self._local_suggestions(report)
        if save:
            save_history(report)
        return report

    def _collect_evidence(self, citation: Citation) -> VerificationEvidence:
        """Query every external source concurrently via the shared pool.

        Isolated source outages (for example a Semantic Scholar rate-limit)
        should not dilute clean misses from the rest of the enabled sources.
        ``lookup_failed`` is reserved for broad outages where unavailable
        sources outnumber successful source attempts.
        """
        futures = {
            spec.name: _EVIDENCE_EXECUTOR.submit(self._lookup_source, spec, citation)
            for spec in _SOURCE_SPECS
        }
        records: dict[str, Citation | None] = {}
        failed_count = 0
        completed_count = 0
        for name, future in futures.items():
            try:
                records[name] = future.result()
                completed_count += 1
            except LookupUnavailable:
                records[name] = None
                failed_count += 1
        lookup_failed = failed_count >= 3 and failed_count > completed_count
        return VerificationEvidence(**records, lookup_failed=lookup_failed)

    def _lookup_source(self, spec: SourceSpec, citation: Citation) -> Citation | None:
        client = getattr(self, spec.client_attr)
        if spec.arxiv_id_lookup and citation.arxiv_id:
            return client.by_id(citation.arxiv_id)
        if citation.doi and spec.skip_title_when_doi:
            return None
        if citation.doi and spec.doi_lookup:
            hit = client.by_doi(citation.doi)
            if hit or not spec.fallback_after_doi_miss:
                return hit
        if not spec.title_lookup:
            return None
        if spec.title_accepts_year:
            return client.search_by_title(citation.title, citation.year)
        return client.search_by_title(citation.title)

    @staticmethod
    def _local_suggestions(report: VerificationReport) -> list[str]:
        failed = [r for r in report.rule_results if not r.passed]
        if not failed:
            return ["规则证据链基本一致，可作为可信引用保留。"]
        return [f"{r.name}：{r.reason}" for r in failed[:4]]
