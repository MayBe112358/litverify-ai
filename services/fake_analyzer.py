"""Fake citation pattern analysis for task one."""
from __future__ import annotations

import re

import pandas as pd


NEGATIVE_RULE_HINTS = {
    "DOI": "DOI 伪造或不可解析",
    "标题": "标题与权威库不一致",
    "作者": "作者列表错配",
    "期刊": "来源期刊/会议错配",
    "年份": "年份错配",
    "卷": "卷期页信息错配",
    "ArXiv": "ArXiv ID 伪造",
    "CrossRef": "跨库证据不足",
}

# ``rule_name:score`` pairs the batch verifier writes into the "reasons" column.
# We treat any rule below 0.5 as a real failure when looking for patterns —
# substring checks like ":0." would also match ``title_match:0.95``.
_RULE_TOKEN_PATTERN = re.compile(r"([^|:]+?):(\d*\.?\d+)")
_FAIL_THRESHOLD = 0.5
_FAKE_LIKE = {"FAKE", "SUSPICIOUS"}
# Task-one grouping dimensions → the column names that may hold them.
_GROUP_DIMENSIONS = {
    "模型": ("生成模型", "模型", "model"),
    "领域": ("学术领域", "领域", "field", "discipline"),
    "主题": ("有关主题", "主题", "topic"),
}


def _verdict_series(df: pd.DataFrame) -> pd.Series:
    """Return an English-verdict series, mapping the Chinese 验证结果 if needed."""
    if "verdict" in df.columns:
        return df["verdict"].fillna("UNKNOWN")
    if "验证结果" in df.columns:
        back = {"可信": "REAL", "可疑": "SUSPICIOUS", "虚假": "FAKE", "错误": "ERROR"}
        return df["验证结果"].map(lambda v: back.get(str(v), "UNKNOWN")).fillna("UNKNOWN")
    return pd.Series(dtype=object)


def verdict_counts(df: pd.DataFrame) -> pd.Series:
    """Count REAL/SUSPICIOUS/FAKE verdicts."""
    series = _verdict_series(df)
    return series.value_counts() if not series.empty else pd.Series(dtype=int)


def rule_failure_profile(df: pd.DataFrame) -> pd.DataFrame:
    """Extract high-frequency failure types from the reasons column."""
    if "reasons" not in df.columns:
        return pd.DataFrame(columns=["pattern", "count"])
    counts: dict[str, int] = {}
    for value in df["reasons"].fillna("").astype(str):
        failed_names = {
            name.strip()
            for name, score in _RULE_TOKEN_PATTERN.findall(value)
            if _safe_float(score) < _FAIL_THRESHOLD
        }
        for name in failed_names:
            for key, label in NEGATIVE_RULE_HINTS.items():
                if key in name:
                    counts[label] = counts.get(label, 0) + 1
                    break
    return (
        pd.DataFrame([{"pattern": k, "count": v} for k, v in counts.items()])
        .sort_values("count", ascending=False)
        if counts
        else pd.DataFrame(columns=["pattern", "count"])
    )


def _safe_float(value: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 1.0


def group_fake_rates(df: pd.DataFrame) -> dict[str, list[dict]]:
    """Per-dimension fake rates for task one.

    For each available dimension (生成模型 / 学术领域 / 有关主题) return a row
    per group with total, fake-like count, fake rate and the single most
    common failing rule — exactly the "各模型虚假文献率" the contest asks for.
    """
    verdicts = _verdict_series(df)
    if verdicts.empty:
        return {}
    work = df.copy()
    work["_verdict"] = verdicts
    out: dict[str, list[dict]] = {}
    for label, candidates in _GROUP_DIMENSIONS.items():
        col = next((c for c in candidates if c in work.columns), None)
        if col is None:
            continue
        rows = []
        for name, sub in work.groupby(col):
            total = int(len(sub))
            fake_like = int(sub["_verdict"].isin(_FAKE_LIKE).sum())
            failures = rule_failure_profile(sub)
            rows.append({
                "group": str(name),
                "total": total,
                "fake_like": fake_like,
                "fake_rate": round(fake_like / total, 4) if total else 0,
                "top_failure": failures.iloc[0]["pattern"] if not failures.empty else "—",
            })
        out[label] = sorted(rows, key=lambda r: r["fake_rate"], reverse=True)
    return out


def build_fake_pattern_report(df: pd.DataFrame) -> dict:
    """Return deterministic task-one insight data."""
    counts = verdict_counts(df)
    failures = rule_failure_profile(df)
    total = int(len(df))
    fake_like = int(counts.get("FAKE", 0) + counts.get("SUSPICIOUS", 0))
    top_patterns = failures.head(5)["pattern"].tolist() if not failures.empty else []
    return {
        "total": total,
        "fake_like": fake_like,
        "fake_like_ratio": fake_like / total if total else 0,
        "verdict_counts": counts.to_dict(),
        "top_patterns": top_patterns,
        "groups": group_fake_rates(df),
    }
