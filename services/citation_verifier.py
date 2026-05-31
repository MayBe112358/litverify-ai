"""Single-citation verification pipeline."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

from config.settings import settings
from db.history import save_history
from llm.insight_generator import explain_verification
from services.api_errors import LookupUnavailable
from services.arxiv_client import ArxivClient
from services.citation_parser import CitationParser
from services.crossref_client import CrossRefClient
from services.openalex_client import OpenAlexClient
from services.rule_engine import (
    Citation,
    RuleConfig,
    RuleEngine,
    VerificationEvidence,
    VerificationReport,
)


# Module-level executor reused across every CitationVerifier instance and
# every batch row — avoids the overhead of creating/destroying 3 threads
# per evidence collection (200 rows × 3 threads previously meant 600
# thread spin-ups). 12 workers comfortably saturate 3 parallel API
# requests across the default 4-worker batch pool.
_EVIDENCE_EXECUTOR = ThreadPoolExecutor(max_workers=12, thread_name_prefix="lv-evidence")


def _session_flag(key: str, default: bool = True) -> bool:
    try:
        import streamlit as st

        return bool(st.session_state.get(key, default))
    except Exception:
        return default


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
    ) -> None:
        self.parser = parser or CitationParser()
        self.rule_engine = rule_engine or RuleEngine(
            rules=_session_rules(),
            thresholds=_session_thresholds(),
        )
        self.crossref = crossref or CrossRefClient()
        self.openalex = openalex or OpenAlexClient()
        self.arxiv = arxiv or ArxivClient()
        # Read the data-source toggles HERE — __init__ runs on the main
        # Streamlit thread, where st.session_state is reachable. The actual
        # lookups run inside _EVIDENCE_EXECUTOR worker threads (and, for
        # batches, a second pool on top), where st.session_state raises
        # NoSessionContext and _session_flag would silently fall back to its
        # default. Capturing the flags once, up front, is the only place the
        # toggles can be honoured.
        self.crossref_enabled = _session_flag("crossref_enabled", True)
        self.openalex_enabled = _session_flag("openalex_enabled", True)
        self.arxiv_enabled = _session_flag("arxiv_enabled", True)

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
        """Query CrossRef/OpenAlex/arXiv concurrently via the shared pool.

        A :class:`LookupUnavailable` from any source flips ``lookup_failed``
        so the rule engine stays neutral instead of treating an outage as a
        confirmed miss.
        """
        futures = {
            "crossref": _EVIDENCE_EXECUTOR.submit(self._crossref_lookup, citation),
            "openalex": _EVIDENCE_EXECUTOR.submit(self._openalex_lookup, citation),
            "arxiv": _EVIDENCE_EXECUTOR.submit(self._arxiv_lookup, citation),
        }
        records: dict[str, Citation | None] = {}
        lookup_failed = False
        for name, future in futures.items():
            try:
                records[name] = future.result()
            except LookupUnavailable:
                records[name] = None
                lookup_failed = True
        return VerificationEvidence(
            crossref=records["crossref"],
            openalex=records["openalex"],
            arxiv=records["arxiv"],
            lookup_failed=lookup_failed,
        )

    def _crossref_lookup(self, citation):
        if not self.crossref_enabled:
            return None
        if citation.doi:
            # 用户已给出 DOI 是强诉求：如果该 DOI 无法在权威库解析，绝不
            # 退化到标题搜索去捡一个名字接近的论文当"弱证据"——那会把
            # 假 DOI 的可疑度稀释掉，让 FAKE 被误判为 SUSPICIOUS。
            return self.crossref.by_doi(citation.doi)
        return self.crossref.search_by_title(citation.title, citation.year)

    def _openalex_lookup(self, citation):
        if not self.openalex_enabled:
            return None
        if citation.doi:
            return self.openalex.by_doi(citation.doi)
        return self.openalex.search_by_title(citation.title, citation.year)

    def _arxiv_lookup(self, citation):
        if not self.arxiv_enabled:
            return None
        if citation.arxiv_id:
            return self.arxiv.by_id(citation.arxiv_id)
        # 没有 arXiv ID 时只用标题搜索 —— 但这里更保守：只在用户没给 DOI
        # 时才走，避免给 DOI 的样本同时被三个来源的 fallback 命中无关篇目。
        if citation.doi:
            return None
        return self.arxiv.search_by_title(citation.title)

    @staticmethod
    def _local_suggestions(report: VerificationReport) -> list[str]:
        failed = [r for r in report.rule_results if not r.passed]
        if not failed:
            return ["规则证据链基本一致，可作为可信引用保留。"]
        return [f"{r.name}：{r.reason}" for r in failed[:4]]
