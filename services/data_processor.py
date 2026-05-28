"""Concurrent batch citation verification."""
from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, Mapping

import pandas as pd

from services.citation_verifier import CitationVerifier
from services.rule_engine import Citation
from utils.doi_utils import extract_doi


ProgressCallback = Callable[[int, int, dict], None]

# Authors might be packed as "A, B; C / D、E、F". Split on every common
# separator so the rule engine sees a real list.
_AUTHOR_SPLIT = re.compile(r"\s*[,;；、，/]\s*|\s+and\s+", re.IGNORECASE)


def batch_verify(
    df: pd.DataFrame,
    citation_col: str,
    max_workers: int = 8,
    on_progress: ProgressCallback | None = None,
    save_history: bool = False,
) -> pd.DataFrame:
    """Verify a DataFrame column and return the original rows plus results.

    ``save_history`` defaults to ``False`` so a 200-row batch doesn't
    flood the SQLite history table; flip it on when the caller wants the
    rows persisted alongside single-citation runs.
    """
    verifier = CitationVerifier()
    rows = df.to_dict("records")

    def _job(index: int, row: dict[str, Any]) -> tuple[int, dict]:
        return index, _run_one(
            lambda: verifier.verify(str(row.get(citation_col, "")), with_llm_explain=False, save=save_history),
            _english_ok,
            _english_err,
        )

    return _dispatch_jobs(df, rows, _job, max_workers, on_progress)


def batch_verify_structured(
    df: pd.DataFrame,
    column_map: Mapping[str, str],
    max_workers: int = 8,
    on_progress: ProgressCallback | None = None,
    save_history: bool = False,
) -> pd.DataFrame:
    """Verify a spreadsheet whose fields are already split across columns.

    ``column_map`` maps logical citation roles ("title" / "authors" /
    "year" / "venue" / "doi") to the DataFrame column that holds each
    role. Missing keys simply mean that field stays empty on the
    constructed :class:`Citation`. This path skips the heuristic parser
    entirely — important for inputs like the contest test set where
    title and authors are already cleanly separated.
    """
    verifier = CitationVerifier()
    rows = df.to_dict("records")

    def _job(index: int, row: dict[str, Any]) -> tuple[int, dict]:
        citation = _build_structured_citation(row, column_map)
        return index, _run_one(
            lambda: verifier.verify_citation(citation, with_llm_explain=False, save=save_history),
            _structured_ok,
            _structured_err,
        )

    return _dispatch_jobs(df, rows, _job, max_workers, on_progress)


def _dispatch_jobs(
    df: pd.DataFrame,
    rows: list[dict[str, Any]],
    job: Callable[[int, dict[str, Any]], tuple[int, dict]],
    max_workers: int,
    on_progress: ProgressCallback | None,
) -> pd.DataFrame:
    results: list[dict | None] = [None] * len(rows)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(job, index, row) for index, row in enumerate(rows)]
        for done, future in enumerate(as_completed(futures), 1):
            index, payload = future.result()
            results[index] = payload
            if on_progress:
                on_progress(done, len(rows), payload)
    result_df = pd.DataFrame(results)
    # Drop pre-existing columns that our result overwrites (e.g. the contest
    # Excel's empty 验证结果 / 虚假特征) so they're filled in place rather than
    # duplicated alongside the new ones.
    base = df.reset_index(drop=True).drop(
        columns=[c for c in result_df.columns if c in df.columns], errors="ignore"
    )
    return pd.concat([base, result_df], axis=1)


# Verdict labels for the Chinese-facing result columns.
_VERDICT_ZH = {"REAL": "可信", "SUSPICIOUS": "可疑", "FAKE": "虚假", "ERROR": "错误"}


def _run_one(
    work: Callable[[], Any],
    on_ok: Callable[[Any], dict[str, Any]],
    on_err: Callable[[Exception], dict[str, Any]],
) -> dict[str, Any]:
    try:
        return on_ok(work())
    except Exception as exc:  # noqa: BLE001 - surfaced into the result row
        return on_err(exc)


def _rule_scores(report) -> str:
    return " | ".join(f"{item.name}:{item.score:.2f}" for item in report.rule_results)


def _english_ok(report) -> dict[str, Any]:
    evidence = report.evidence.best_record()
    return {
        "verdict": report.verdict,
        "score": report.overall_score,
        "matched_doi": evidence.doi if evidence else None,
        "matched_title": evidence.title if evidence else None,
        "reasons": _rule_scores(report),
        "suggestions": " | ".join(report.suggestions),
    }


def _english_err(exc: Exception) -> dict[str, Any]:
    return {
        "verdict": "ERROR",
        "score": 0.0,
        "matched_doi": None,
        "matched_title": None,
        "reasons": str(exc),
        "suggestions": "请检查原始引用格式或网络状态。",
    }


def _structured_ok(report) -> dict[str, Any]:
    """Chinese-facing result columns + the machine columns the fake-analysis /
    export paths still rely on (verdict / score / reasons)."""
    evidence = report.evidence.best_record()
    failed = [r.name for r in report.rule_results if r.score < 0.5]
    return {
        "验证结果": _VERDICT_ZH.get(report.verdict, report.verdict),
        "可信度分数": report.overall_score,
        "虚假特征": "；".join(failed) if failed else "无明显问题",
        "命中DOI": evidence.doi if evidence else None,
        "命中标题": evidence.title if evidence else None,
        "verdict": report.verdict,
        "score": report.overall_score,
        "reasons": _rule_scores(report),
    }


def _structured_err(exc: Exception) -> dict[str, Any]:
    return {
        "验证结果": "错误",
        "可信度分数": 0.0,
        "虚假特征": str(exc),
        "命中DOI": None,
        "命中标题": None,
        "verdict": "ERROR",
        "score": 0.0,
        "reasons": str(exc),
    }


def _build_structured_citation(
    row: Mapping[str, Any],
    column_map: Mapping[str, str],
) -> Citation:
    """Read each role column from ``row`` and produce a :class:`Citation`."""
    title = _cell(row, column_map.get("title"))
    venue = _cell(row, column_map.get("venue"))
    doi = extract_doi(_cell(row, column_map.get("doi"))) or _cell(row, column_map.get("doi")) or None
    if doi:
        doi = doi.lower()
    year_raw = _cell(row, column_map.get("year"))
    year = _safe_year(year_raw)
    authors = _split_authors(_cell(row, column_map.get("authors")))
    raw_repr = " | ".join(filter(None, [title, ";".join(authors) if authors else None, venue, str(year) if year else None, doi]))
    return Citation(
        title=title or None,
        authors=authors,
        year=year,
        venue=venue or None,
        doi=doi,
        raw=raw_repr or None,
    )


def _cell(row: Mapping[str, Any], col: str | None) -> str:
    if not col or col not in row:
        return ""
    value = row[col]
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value).strip()


def _safe_year(value: str) -> int | None:
    if not value:
        return None
    match = re.search(r"(18|19|20|21)\d{2}", value)
    if not match:
        return None
    try:
        return int(match.group(0))
    except ValueError:
        return None


def _split_authors(value: str) -> list[str]:
    if not value:
        return []
    parts = [p.strip(" .") for p in _AUTHOR_SPLIT.split(value) if p.strip(" .")]
    # Keep at most 12 authors to match the heuristic parser's output shape.
    return parts[:12]
