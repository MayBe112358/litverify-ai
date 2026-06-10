"""Tests for the round-2 improvements: structured columns, Chinese output,
year boundary, arXiv recognition, API error/miss distinction, fake grouping."""
from __future__ import annotations

import pandas as pd
import pytest
import requests

from services.agent_router import _detect_structured_columns
from utils.dataframe import df_to_json_safe_records
from services.api_errors import is_transient_error
from services.data_processor import _build_structured_citation, _split_authors
from services.fake_analyzer import group_fake_rates, rule_failure_profile, verdict_counts
from services.rule_engine import Citation, RuleEngine, VerificationEvidence
from utils.doi_utils import extract_arxiv


# --- ③ arXiv unified recognition ---------------------------------------- #
@pytest.mark.parametrize(
    "text,expected",
    [
        ("arXiv:1706.03762v5", "1706.03762v5"),
        ("https://arxiv.org/abs/1706.03762", "1706.03762"),
        ("10.48550/arXiv.1706.03762", "1706.03762"),
        ("doi:10.11897/COMPJ.2025.0012", None),  # DOI tail must not be misread
        ("astro-ph/0211000", "astro-ph/0211000"),
    ],
)
def test_extract_arxiv_variants(text, expected):
    assert extract_arxiv(text) == expected


# --- ① year boundary 1900-2026 ------------------------------------------ #
@pytest.mark.parametrize("year,should_pass", [(1899, False), (1900, True), (2026, True), (2027, False)])
def test_year_boundary(year, should_pass):
    engine = RuleEngine()
    report = engine.evaluate(Citation(title="some test paper", year=year), VerificationEvidence())
    rule = next(r for r in report.rule_results if r.id == "year_reasonable")
    assert rule.passed is should_pass


def test_local_metadata_rules_present():
    engine = RuleEngine()
    ids = {r.id for r in engine.evaluate(Citation(title="x paper"), VerificationEvidence()).rule_results}
    assert {"author_format", "venue_format", "title_quality"} <= ids


# --- ② API error vs miss ------------------------------------------------- #
def test_is_transient_error():
    timeout = requests.Timeout("boom")
    assert is_transient_error(timeout) is True

    resp404 = requests.Response()
    resp404.status_code = 404
    err404 = requests.HTTPError(response=resp404)
    assert is_transient_error(err404) is False

    resp503 = requests.Response()
    resp503.status_code = 503
    err503 = requests.HTTPError(response=resp503)
    assert is_transient_error(err503) is True


def test_lookup_failed_stays_neutral_vs_miss():
    """A transient outage must not score lower than a confirmed miss."""
    engine = RuleEngine()
    user = Citation(title="Deep learning", authors=["LeCun Y"], year=2015, doi="10.1038/nature14539")
    outage = engine.evaluate(user, VerificationEvidence(lookup_failed=True))
    miss = engine.evaluate(user, VerificationEvidence(lookup_failed=False))
    assert outage.overall_score > miss.overall_score


# --- C/④ structured columns + Chinese output ----------------------------- #
def _contest_df():
    return pd.DataFrame(
        {
            "生成模型": ["豆包"],
            "学术领域": ["计算机科学"],
            "有关主题": ["深度学习"],
            "完整标题": ["一个测试标题"],
            "作者姓名": ["王宇轩，李泽峰"],
            "发表的期刊 / 会议全称": ["计算机学报"],
            "发表年份": [2025],
            "完整的 DOI 编号": ["10.99999/fake.cn"],
            "验证结果": [None],
            "虚假特征": [None],
        }
    )


def test_detect_structured_columns():
    mapping = _detect_structured_columns(_contest_df())
    assert mapping is not None
    assert mapping["title"] == "完整标题"
    assert mapping["doi"] == "完整的 DOI 编号"
    assert mapping["authors"] == "作者姓名"


def test_detect_structured_columns_single_column_returns_none():
    df = pd.DataFrame({"citation_text": ["A. Some paper. Journal, 2020."]})
    assert _detect_structured_columns(df) is None


def test_build_structured_citation_splits_authors_and_year():
    row = {
        "完整标题": "一个测试标题",
        "作者姓名": "王宇轩，李泽峰，陈星燃",
        "发表的期刊 / 会议全称": "计算机学报",
        "发表年份": 2025,
        "完整的 DOI 编号": "10.99999/fake.cn",
    }
    mapping = {"title": "完整标题", "authors": "作者姓名", "venue": "发表的期刊 / 会议全称",
               "year": "发表年份", "doi": "完整的 DOI 编号"}
    cite = _build_structured_citation(row, mapping)
    assert cite.title == "一个测试标题"
    assert cite.authors == ["王宇轩", "李泽峰", "陈星燃"]
    assert cite.year == 2025
    assert cite.doi == "10.99999/fake.cn"


def test_split_authors_multiple_separators():
    assert _split_authors("A, B; C、D and E") == ["A", "B", "C", "D", "E"]


def test_df_to_records_nan_to_none():
    df = pd.DataFrame({"a": [1, None], "report_json": ["x", "y"]})
    records = df_to_json_safe_records(df)
    assert "report_json" not in records[0]
    assert records[1]["a"] is None


# --- ⑤ fake grouping ----------------------------------------------------- #
def test_group_fake_rates_by_model():
    df = pd.DataFrame(
        {
            "生成模型": ["豆包", "豆包", "DeepSeek"],
            "verdict": ["FAKE", "REAL", "FAKE"],
            "reasons": ["DOI 解析:0.00", "DOI 解析:1.00", "DOI 解析:0.00"],
        }
    )
    groups = group_fake_rates(df)
    assert "模型" in groups
    by_model = {row["group"]: row for row in groups["模型"]}
    assert by_model["豆包"]["fake_rate"] == 0.5
    assert by_model["DeepSeek"]["fake_rate"] == 1.0


def test_verdict_counts_maps_chinese_column():
    df = pd.DataFrame({"验证结果": ["可信", "虚假", "虚假"]})
    counts = verdict_counts(df)
    assert counts.get("FAKE") == 2
    assert counts.get("REAL") == 1


def test_rule_failure_profile_uses_numeric_threshold():
    # title_match:0.95 must NOT be counted as a failure despite containing ':0.'
    df = pd.DataFrame({"reasons": ["标题与权威库一致:0.95 | DOI 在 CrossRef/OpenAlex 可解析:0.00"]})
    profile = rule_failure_profile(df)
    patterns = set(profile["pattern"]) if not profile.empty else set()
    assert "DOI 伪造或不可解析" in patterns
    assert "标题与权威库不一致" not in patterns
