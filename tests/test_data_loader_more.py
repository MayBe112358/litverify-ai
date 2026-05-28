from __future__ import annotations

import io

import pandas as pd

from services.data_loader import _read_json_smart, load_dataframe


def test_read_json_smart_array() -> None:
    raw = b'[{"city":"hz","sales":100},{"city":"sh","sales":200}]'
    df = _read_json_smart(raw)
    assert df["sales"].sum() == 300


def test_read_json_smart_jsonlines() -> None:
    raw = b'{"city":"hz","sales":100}\n{"city":"sh","sales":200}\n'
    df = _read_json_smart(raw)
    assert df["sales"].sum() == 300


def test_load_dataframe_handles_tsv() -> None:
    raw = "city\tsales\nhz\t100\nsh\t200\n".encode("utf-8")
    df = load_dataframe(io.BytesIO(raw), "data.tsv")
    assert list(df.columns) == ["city", "sales"]
    assert int(df["sales"].sum()) == 300


def test_load_dataframe_handles_excel(tmp_path) -> None:
    file = tmp_path / "sales.xlsx"
    pd.DataFrame({"city": ["hz", "sh"], "sales": [100, 200]}).to_excel(file, index=False)
    df = load_dataframe(file, file.name)
    assert int(df["sales"].sum()) == 300


def test_load_dataframe_handles_parquet(tmp_path) -> None:
    file = tmp_path / "sales.parquet"
    pd.DataFrame({"city": ["hz", "sh"], "sales": [100, 200]}).to_parquet(file, index=False)
    df = load_dataframe(file, file.name)
    assert int(df["sales"].sum()) == 300
