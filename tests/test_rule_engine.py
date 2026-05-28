from __future__ import annotations

from services.rule_engine import Citation, RuleEngine, VerificationEvidence


def test_rule_engine_real_when_fields_match() -> None:
    user = Citation(
        title="Attention is all you need",
        authors=["Ashish Vaswani", "Noam Shazeer"],
        year=2017,
        venue="NeurIPS",
        doi="10.48550/arxiv.1706.03762",
    )
    record = Citation(
        title="Attention is all you need",
        authors=["Ashish Vaswani", "Noam Shazeer"],
        year=2017,
        venue="NeurIPS",
        doi="10.48550/arxiv.1706.03762",
        source="CrossRef",
    )
    report = RuleEngine().evaluate(user, VerificationEvidence(crossref=record, openalex=record))
    assert report.verdict == "REAL"
    assert report.overall_score >= 80


def test_rule_engine_fake_when_doi_and_title_conflict() -> None:
    user = Citation(
        title="A Fabricated Transformer Paper",
        authors=["Nobody"],
        year=2023,
        venue="Nature Machine Intelligence",
        doi="10.9999/fake",
    )
    record = Citation(
        title="Completely Different Work",
        authors=["Somebody Else"],
        year=2020,
        venue="Different Venue",
        doi="10.1111/real",
        source="CrossRef",
    )
    report = RuleEngine().evaluate(user, VerificationEvidence(crossref=record))
    assert report.verdict in {"FAKE", "SUSPICIOUS"}
    assert any(item.id == "title_match" and item.score < 0.8 for item in report.rule_results)
