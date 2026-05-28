from __future__ import annotations

from services.crossref_client import CrossRefClient


def test_crossref_to_citation_maps_fields() -> None:
    item = {
        "title": ["A Test Paper"],
        "author": [{"given": "Ada", "family": "Lovelace"}],
        "issued": {"date-parts": [[2024]]},
        "container-title": ["Journal of Tests"],
        "DOI": "10.1234/ABC",
        "type": "journal-article",
    }
    citation = CrossRefClient._to_citation(item)
    assert citation.title == "A Test Paper"
    assert citation.authors == ["Ada Lovelace"]
    assert citation.year == 2024
    assert citation.doi == "10.1234/abc"
