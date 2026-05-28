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
