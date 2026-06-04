from __future__ import annotations

from services.agent_router import report_from_dict
from services.api_errors import LookupUnavailable
from services.citation_verifier import CitationVerifier
from services.datacite_client import DataCiteClient
from services.dblp_client import _to_citation as dblp_to_citation
from services.pubmed_client import PubMedClient
from services.rule_engine import Citation, RuleEngine, VerificationEvidence
from services.semantic_scholar_client import SemanticScholarClient
from services.wanfang_client import _to_citation as wanfang_to_citation


def test_pubmed_summary_payload_to_citation() -> None:
    citation = PubMedClient._to_citation(
        {
            "title": "Clinical decision support with machine learning.",
            "authors": [{"name": "Ada Lovelace"}, {"name": "Grace Hopper"}],
            "pubdate": "2024 Jan",
            "fulljournalname": "Journal of Medical AI",
            "articleids": [{"idtype": "doi", "value": "10.1000/test"}],
            "volume": "12",
            "issue": "1",
            "pages": "1-9",
            "pubtype": ["Journal Article"],
        },
        uid="12345",
    )
    assert citation.source == "PubMed"
    assert citation.doi == "10.1000/test"
    assert citation.year == 2024
    assert citation.authors == ["Ada Lovelace", "Grace Hopper"]
    assert citation.url == "https://pubmed.ncbi.nlm.nih.gov/12345/"


def test_semantic_scholar_payload_to_citation() -> None:
    citation = SemanticScholarClient._to_citation(
        {
            "title": "Attention is all you need",
            "authors": [{"name": "Ashish Vaswani"}],
            "year": 2017,
            "venue": "NeurIPS",
            "externalIds": {"DOI": "10.48550/arxiv.1706.03762"},
            "citationCount": 100000,
            "publicationTypes": ["Conference"],
            "url": "https://www.semanticscholar.org/paper/test",
        }
    )
    assert citation.source == "Semantic Scholar"
    assert citation.doi == "10.48550/arxiv.1706.03762"
    assert citation.venue == "NeurIPS"
    assert citation.citation_count == 100000


def test_dblp_payload_to_citation() -> None:
    citation = dblp_to_citation(
        {
            "title": "A scalable citation verifier.",
            "authors": {"author": [{"text": "Leslie Lamport"}, {"text": "Barbara Liskov"}]},
            "year": "2025",
            "venue": "SIGMOD",
            "doi": "10.1145/example",
            "url": "https://dblp.org/rec/conf/test/example",
            "type": "Conference and Workshop Papers",
        }
    )
    assert citation.source == "DBLP"
    assert citation.title == "A scalable citation verifier"
    assert citation.authors == ["Leslie Lamport", "Barbara Liskov"]
    assert citation.year == 2025


def test_datacite_payload_to_citation() -> None:
    citation = DataCiteClient._to_citation(
        {
            "titles": [{"title": "Data package for citation verification"}],
            "creators": [{"givenName": "Katherine", "familyName": "Johnson"}],
            "publicationYear": 2026,
            "publisher": "Example Repository",
            "doi": "10.14454/qdd3-ps68",
            "url": "https://example.org/dataset",
            "types": {"resourceTypeGeneral": "Dataset"},
        }
    )
    assert citation.source == "DataCite"
    assert citation.type == "Dataset"
    assert citation.authors == ["Katherine Johnson"]
    assert citation.doi == "10.14454/qdd3-ps68"


def test_wanfang_payload_to_citation() -> None:
    citation = wanfang_to_citation(
        {
            "title": "Citation verification in Chinese journals",
            "creators": ["Qian Xuesen", "Tu Youyou"],
            "publishYear": "2023",
            "periodicalTitle": "Chinese Journal of Data Quality",
            "doi": "10.1234/cjdq.2023.001",
            "volume": "8",
            "issue": "2",
            "page": "20-28",
            "citedCount": "16",
        }
    )
    assert citation.source == "Wanfang"
    assert citation.year == 2023
    assert citation.venue == "Chinese Journal of Data Quality"
    assert citation.citation_count == 16


def test_rule_engine_counts_non_crossref_doi_resolver() -> None:
    user = Citation(title="A dataset", doi="10.14454/qdd3-ps68")
    datacite = Citation(
        title="A dataset",
        doi="10.14454/qdd3-ps68",
        source="DataCite",
    )
    report = RuleEngine().evaluate(user, VerificationEvidence(datacite=datacite))
    doi_rule = next(item for item in report.rule_results if item.id == "doi_resolve")
    assert doi_rule.score == 1.0
    assert "DataCite" in doi_rule.reason


def test_report_from_dict_restores_all_evidence_sources() -> None:
    payload = {
        "user_citation": {"title": "Attention is all you need"},
        "evidence": {
            "semantic_scholar": {
                "title": "Attention is all you need",
                "doi": "10.48550/arxiv.1706.03762",
                "source": "Semantic Scholar",
            },
            "datacite": {
                "doi": "10.48550/arxiv.1706.03762",
                "source": "DataCite",
            },
            "lookup_failed": True,
        },
        "rule_results": [],
        "overall_score": 88,
        "verdict": "REAL",
    }
    report = report_from_dict(payload)
    assert report.evidence.semantic_scholar is not None
    assert report.evidence.datacite is not None
    assert report.evidence.lookup_failed is True


def test_single_source_failure_does_not_mark_global_lookup_failed(monkeypatch) -> None:
    verifier = CitationVerifier()
    called_sources: list[str] = []

    def lookup(spec, _citation):
        called_sources.append(spec.name)
        if spec.name == "semantic_scholar":
            raise LookupUnavailable("rate limited")
        return None

    monkeypatch.setattr(verifier, "_lookup_source", lookup)

    evidence = verifier._collect_evidence(Citation(title="Fabricated citation"))
    assert evidence.lookup_failed is False
    assert set(called_sources) == {
        "crossref",
        "openalex",
        "arxiv",
        "pubmed",
        "semantic_scholar",
        "dblp",
        "wanfang",
        "datacite",
        "doidb",
    }
