"""Multi-format data loading and profiling utilities."""
from __future__ import annotations

import io
import tempfile
from pathlib import Path
from typing import BinaryIO

import pandas as pd

from config.settings import settings

try:  # Optional at import time so the app can still boot in lean environments.
    import chardet
except Exception:  # pragma: no cover - depends on environment
    chardet = None  # type: ignore[assignment]

try:  # DuckDB is only required for large CSV streaming.
    import duckdb
except Exception:  # pragma: no cover - depends on environment
    duckdb = None  # type: ignore[assignment]


SUPPORTED_FORMATS = {".csv", ".tsv", ".xlsx", ".xls", ".json", ".parquet"}


def detect_encoding(raw: bytes, sample: int = 10000) -> str:
    """Detect text encoding from raw bytes."""
    if chardet is not None:
        guess = chardet.detect(raw[:sample])
        return guess.get("encoding") or "utf-8"
    for encoding in ("utf-8-sig", "utf-8", "gb18030", "gbk"):
        try:
            raw[:sample].decode(encoding)
            return encoding
        except UnicodeDecodeError:
            continue
    return "utf-8"


def _read_raw(file_obj: BinaryIO | str | Path) -> bytes:
    if hasattr(file_obj, "read"):
        return file_obj.read()
    return Path(file_obj).read_bytes()


def _read_json_smart(raw: bytes) -> pd.DataFrame:
    """Read JSON or JSON Lines, whichever the file actually is."""
    head = raw[:2048].lstrip()
    # Heuristic: a JSON array/object opens with '[' or '{'. If the first
    # non-whitespace char is '{' AND there's a newline before any closing
    # bracket, it's most likely jsonlines.
    if head.startswith(b"{") and b"\n" in raw[: min(len(raw), 4096)]:
        try:
            return pd.read_json(io.BytesIO(raw), lines=True)
        except ValueError:
            pass
    return pd.read_json(io.BytesIO(raw))


def _read_large_csv_with_duckdb(raw: bytes, filename: str, sep: str) -> pd.DataFrame:
    if duckdb is None:
        raise RuntimeError("当前环境未安装 duckdb，无法启用大文件流式读取。")
    suffix = Path(filename).suffix or ".csv"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(raw)
        tmp_path = Path(tmp.name)
    try:
        # Use parameterized SQL so a tab separator (sep == "\t") survives
        # transit without being interpreted or escaped by string interpolation.
        connection = duckdb.connect()
        try:
            return connection.execute(
                "SELECT * FROM read_csv_auto(?, delim=?, sample_size=200000)",
                [tmp_path.as_posix(), sep],
            ).fetchdf()
        finally:
            connection.close()
    finally:
        tmp_path.unlink(missing_ok=True)


def infer_datetime_columns(df: pd.DataFrame, max_unique_ratio: float = 0.98) -> pd.DataFrame:
    """Infer likely datetime columns without forcing high-cardinality identifiers."""
    result = df.copy()
    candidate_cols = [
        col
        for col in result.columns
        if pd.api.types.is_object_dtype(result[col]) or pd.api.types.is_string_dtype(result[col])
    ]
    for col in candidate_cols:
        series = result[col].dropna()
        if series.empty:
            continue
        sample = series.astype(str).head(30)
        looks_temporal = sample.str.contains(r"\d{4}[-/年]\d{1,2}|:\d{2}", regex=True).mean() >= 0.5
        unique_ratio = series.nunique() / len(series)
        has_temporal_name = any(k in col.lower() for k in ["date", "time", "日期", "时间"])
        if not looks_temporal and not has_temporal_name:
            continue
        if unique_ratio > max_unique_ratio and not has_temporal_name:
            continue
        converted = pd.to_datetime(result[col], errors="coerce")
        if converted.notna().mean() >= 0.8:
            result[col] = converted
    return result


def load_dataframe(
    file_obj: BinaryIO | str | Path,
    filename: str | None = None,
    big_file_threshold_mb: int | None = None,
) -> pd.DataFrame:
    """Load CSV, TSV, Excel, JSON or Parquet into a DataFrame."""
    if big_file_threshold_mb is None:
        big_file_threshold_mb = settings.duckdb_threshold_mb
    name = filename or getattr(file_obj, "name", str(file_obj))
    ext = Path(name).suffix.lower()
    if ext not in SUPPORTED_FORMATS:
        raise ValueError(f"暂不支持的文件格式：{ext}")

    raw = _read_raw(file_obj)
    size_mb = len(raw) / (1024 * 1024)

    if ext in {".csv", ".tsv"}:
        sep = "\t" if ext == ".tsv" else ","
        if size_mb > big_file_threshold_mb:
            df = _read_large_csv_with_duckdb(raw, name, sep)
        else:
            encoding = detect_encoding(raw)
            df = pd.read_csv(io.BytesIO(raw), encoding=encoding, sep=sep, low_memory=False)
        return infer_datetime_columns(df)

    if ext in {".xlsx", ".xls"}:
        return infer_datetime_columns(pd.read_excel(io.BytesIO(raw)))

    if ext == ".json":
        return infer_datetime_columns(_read_json_smart(raw))

    if ext == ".parquet":
        return pd.read_parquet(io.BytesIO(raw))

    raise ValueError(f"未实现的格式处理：{ext}")
