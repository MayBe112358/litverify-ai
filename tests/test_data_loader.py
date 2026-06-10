from __future__ import annotations

import io

from services.data_loader import detect_encoding, load_dataframe


def test_detect_encoding_defaults_to_guessable_encoding() -> None:
    assert detect_encoding("城市,销售额\n杭州,100\n".encode("utf-8"))


def test_load_csv_from_bytes() -> None:
    raw = "城市,销售额\n杭州,100\n上海,200\n".encode("utf-8")
    df = load_dataframe(io.BytesIO(raw), "sales.csv")
    assert list(df.columns) == ["城市", "销售额"]
    assert df["销售额"].sum() == 300
