"""Tests for the sandboxed execution of DeepSeek-generated chart code."""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import pytest

from services.agent_router import _exec_chart_code, _heuristic_chart_spec, _run_chart


@pytest.fixture()
def result_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "verdict": ["REAL", "FAKE", "SUSPICIOUS", "REAL"],
            "score": [95.0, 20.5, 55.0, 88.0],
            "生成模型": ["豆包", "DeepSeek", "豆包", "Kimi"],
        }
    )


def test_exec_returns_plotly_figure(result_df: pd.DataFrame) -> None:
    code = (
        "counts = df['verdict'].value_counts().reset_index()\n"
        "counts.columns = ['判定', '数量']\n"
        "fig = px.bar(counts, x='判定', y='数量')\n"
        "fig.update_layout(title='判定分布')\n"
    )
    fig = _exec_chart_code(code, result_df)
    assert isinstance(fig, go.Figure)
    assert fig.layout.title.text == "判定分布"


def test_exec_supports_graph_objects(result_df: pd.DataFrame) -> None:
    code = (
        "fig = go.Figure(go.Scatterpolar(r=df['score'], theta=df['verdict'], fill='toself'))\n"
        "fig.update_layout(title='得分雷达图')\n"
    )
    fig = _exec_chart_code(code, result_df)
    assert isinstance(fig, go.Figure)


@pytest.mark.parametrize(
    "code",
    [
        "import os\nfig = px.bar(df, x='verdict')",
        "open('x.txt', 'w')",
        "eval('1+1')",
        "while True:\n    pass",
        "__import__('os')",
    ],
)
def test_exec_rejects_forbidden_tokens(result_df: pd.DataFrame, code: str) -> None:
    with pytest.raises(ValueError, match="被禁止"):
        _exec_chart_code(code, result_df)


def test_exec_requires_fig_variable(result_df: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="fig"):
        _exec_chart_code("x = df['score'].mean()", result_df)


def test_exec_requires_plotly_figure(result_df: pd.DataFrame) -> None:
    with pytest.raises(TypeError):
        _exec_chart_code("fig = df['score'].mean()", result_df)


def test_exec_does_not_mutate_source_df(result_df: pd.DataFrame) -> None:
    code = "df['score'] = 0\nfig = px.histogram(df, x='score')"
    _exec_chart_code(code, result_df)
    assert result_df["score"].tolist() == [95.0, 20.5, 55.0, 88.0]


def test_run_chart_falls_back_without_api_key(
    result_df: pd.DataFrame, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No API key → deterministic heuristic spec payload."""
    import services.agent_router as agent_router

    monkeypatch.setattr(agent_router, "runtime_api_key", lambda: "")
    history = [
        {
            "role": "assistant",
            "kind": "verify_batch",
            "data": {"rows": result_df.to_dict("records")},
        }
    ]
    message = _run_chart("画个饼图", files=[], history=history)
    assert message["kind"] == "chart"
    assert message["data"]["spec"]["x"] == "verdict"
    assert "fig_json" not in message["data"]


def test_heuristic_spec_prefers_verdict(result_df: pd.DataFrame) -> None:
    spec = _heuristic_chart_spec(result_df)
    assert spec is not None
    assert spec["chart_type"] == "pie"
    assert spec["x"] == "verdict"
