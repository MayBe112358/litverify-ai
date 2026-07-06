from __future__ import annotations

from services.citation_parser import CitationParser
from utils.doi_utils import extract_arxiv, extract_doi, normalize_doi


def test_extract_doi_normalizes_prefix() -> None:
    assert normalize_doi("https://doi.org/10.1145/3368089.3409749") == "10.1145/3368089.3409749"


def test_extract_doi_from_text() -> None:
    assert extract_doi("doi:10.1038/s41586-020-2649-2.") == "10.1038/s41586-020-2649-2"


def test_extract_arxiv_new_style() -> None:
    assert extract_arxiv("arXiv:1706.03762v5") == "1706.03762v5"


def test_parser_extracts_core_fields() -> None:
    parser = CitationParser(use_llm_fallback=False)
    citation = parser.parse(
        "Vaswani A, Shazeer N, Parmar N. Attention is all you need. "
        "Advances in Neural Information Processing Systems, 2017. doi:10.48550/arXiv.1706.03762"
    )
    assert citation.year == 2017
    assert citation.doi == "10.48550/arxiv.1706.03762"
    assert "Attention" in (citation.title or "")
    assert citation.authors


def test_pure_doi_input_skips_llm_fallback(monkeypatch) -> None:
    """只有 DOI 的输入不得触发 LLM 兜底——LLM 会凭记忆幻觉出标题，
    污染后续与权威库的比对（真实 DOI 被拉成可疑）。"""
    parser = CitationParser(use_llm_fallback=True)

    def _boom(text: str) -> None:
        raise AssertionError("LLM fallback must not run for a DOI-only input")

    monkeypatch.setattr(CitationParser, "_llm_parse", staticmethod(_boom))
    citation = parser.parse("doi:10.1038/nature14539")
    assert citation.doi == "10.1038/nature14539"
    assert citation.title is None
    assert citation.authors == []


def test_pure_arxiv_and_url_inputs_skip_llm_fallback(monkeypatch) -> None:
    parser = CitationParser(use_llm_fallback=True)
    monkeypatch.setattr(
        CitationParser,
        "_llm_parse",
        staticmethod(lambda text: (_ for _ in ()).throw(AssertionError("no LLM"))),
    )
    assert parser.parse("arXiv:1706.03762v5").arxiv_id == "1706.03762v5"
    assert parser.parse("https://doi.org/10.1145/3368089.3409749").doi


def test_llm_fields_must_appear_in_input() -> None:
    """LLM 只允许摘录原文，联想出来的字段必须被丢弃。"""
    raw = "Vaswani A. Attention is all you need. 2017."
    assert CitationParser._grounded(raw, "Attention is all you need")
    assert CitationParser._grounded(raw, "ATTENTION IS ALL YOU NEED!")  # 忽略大小写标点
    assert CitationParser._grounded(raw, "Deep learning") is None  # 幻觉标题
    assert CitationParser._grounded(raw, None) is None
