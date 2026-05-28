from __future__ import annotations

from services.citation_parser import CitationParser
from services.citation_verifier import CitationVerifier
from services.rule_engine import Citation


class FakeCrossRef:
    def by_doi(self, doi):
        return Citation(
            title="Attention is all you need",
            authors=["Ashish Vaswani", "Noam Shazeer"],
            year=2017,
            venue="NeurIPS",
            doi=doi,
            source="CrossRef",
        )

    def search_by_title(self, title, year=None):
        return None


class FakeOpenAlex(FakeCrossRef):
    def by_doi(self, doi):
        c = super().by_doi(doi)
        c.source = "OpenAlex"
        return c


class FakeArxiv:
    def by_id(self, arxiv_id):
        return None

    def search_by_title(self, title):
        return None


def test_verifier_pipeline_with_fake_clients() -> None:
    verifier = CitationVerifier(
        parser=CitationParser(use_llm_fallback=False),
        crossref=FakeCrossRef(),
        openalex=FakeOpenAlex(),
        arxiv=FakeArxiv(),
    )
    report = verifier.verify(
        "Vaswani A, Shazeer N. Attention is all you need. NeurIPS, 2017. doi:10.48550/arXiv.1706.03762",
        with_llm_explain=False,
        save=False,
    )
    assert report.verdict == "REAL"
    assert report.evidence.crossref is not None
