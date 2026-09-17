"""
CSV ingestion and summary statistics.

C4: every one of these malformed inputs previously produced a raw traceback
that took down the whole Streamlit page.
"""

from __future__ import annotations

import pytest

from src.session_manager import (
    DataLoadError,
    _dedupe_columns,
    _sniff_delimiter,
    compute_data_summary,
    parse_csv_bytes,
)

# ── C4: malformed input must raise DataLoadError, not crash ───


def test_empty_bytes_rejected() -> None:
    with pytest.raises(DataLoadError, match="empty"):
        parse_csv_bytes(b"", "empty.csv")


def test_whitespace_only_rejected() -> None:
    with pytest.raises(DataLoadError):
        parse_csv_bytes(b"   \n\n  \n", "blank.csv")


def test_header_with_no_rows_rejected() -> None:
    with pytest.raises(DataLoadError, match="no rows"):
        parse_csv_bytes(b"a,b,c\n", "headers.csv")


def test_ragged_rows_rejected_with_a_readable_message() -> None:
    raw = b"a,b\n1,2\n3,4,5,6,7\n8,9,10,11,12\n"
    with pytest.raises(DataLoadError) as exc:
        parse_csv_bytes(raw, "ragged.csv")
    assert "ragged.csv" in str(exc.value)


def test_oversized_upload_rejected(monkeypatch) -> None:
    import dataclasses

    from src.config import settings

    # Settings are frozen, so swap in a replaced copy rather than mutating.
    small = dataclasses.replace(settings, data=dataclasses.replace(settings.data, max_upload_mb=1))
    monkeypatch.setattr("src.session_manager.settings", small)

    with pytest.raises(DataLoadError, match="exceeds"):
        parse_csv_bytes(b"a,b\n" + b"1,2\n" * 400_000, "huge.csv")


@pytest.mark.parametrize(
    "raw",
    [
        bytes(range(256)) * 10,                       # arbitrary binary
        b"\x89PNG\r\n\x1a\n" + bytes(range(200)) * 5,  # a PNG renamed .csv
        b"PK\x03\x04" + b"\x00\x01\x02\x03" * 100,     # a zip/xlsx renamed .csv
    ],
    ids=["binary", "png", "zip"],
)
def test_binary_upload_is_rejected(raw: bytes) -> None:
    """
    latin-1 decodes any byte sequence, so without an explicit check these
    parsed into a meaningless frame of `Unnamed: 0` columns and every
    subsequent question failed for no visible reason.
    """
    with pytest.raises(DataLoadError, match="text CSV"):
        parse_csv_bytes(raw, "binary.csv")


def test_accented_text_is_not_mistaken_for_binary() -> None:
    """The binary heuristic must not reject legitimate non-ASCII CSVs."""
    raw = "Name,City\nJosé,Köln\nRené,Zürich\n".encode()
    assert len(parse_csv_bytes(raw, "accents.csv")) == 2


def test_missing_file_path_rejected() -> None:
    from src.session_manager import SessionManager

    with pytest.raises(DataLoadError, match="not found"):
        SessionManager.load_df_from_path("no/such/file.csv")


# ── Encoding and delimiter handling ───────────────────────────


def test_utf8_bom_is_stripped() -> None:
    df = parse_csv_bytes("Name,Age\nJosé,30\n".encode("utf-8-sig"), "bom.csv")
    assert list(df.columns) == ["Name", "Age"]
    assert df.loc[0, "Name"] == "José"


def test_latin1_fallback() -> None:
    df = parse_csv_bytes("Name,City\nRené,Köln\n".encode("latin-1"), "latin.csv")
    assert len(df) == 1


@pytest.mark.parametrize("delimiter", [",", ";", "\t", "|"])
def test_delimiters_are_sniffed(delimiter: str) -> None:
    """Excel writes ';' separated .csv files in many locales."""
    raw = f"Name{delimiter}Age\nAlice{delimiter}30\nBob{delimiter}25\n".encode()
    df = parse_csv_bytes(raw, "sep.csv")
    assert list(df.columns) == ["Name", "Age"]
    assert len(df) == 2


def test_sniffer_defaults_to_comma_when_ambiguous() -> None:
    assert _sniff_delimiter("single_column\nvalue\n") == ","


# ── Column hygiene ────────────────────────────────────────────


def test_duplicate_columns_are_renamed() -> None:
    """
    Duplicates make `df['col']` return a DataFrame instead of a Series,
    which breaks generated code in confusing ways.
    """
    df = parse_csv_bytes(b"a,a,b\n1,2,3\n", "dupes.csv")
    assert len(set(df.columns)) == 3
    assert df["a"].ndim == 1


def test_column_whitespace_is_stripped() -> None:
    df = parse_csv_bytes(b" Name , Age \nAlice,30\n", "spaces.csv")
    assert list(df.columns) == ["Name", "Age"]


def test_dedupe_is_a_noop_for_unique_columns() -> None:
    import pandas as pd

    df = pd.DataFrame({"a": [1], "b": [2]})
    assert list(_dedupe_columns(df).columns) == ["a", "b"]


# ── Summary statistics ────────────────────────────────────────


def test_summary_reports_shape_and_types(sample_df) -> None:
    summary = compute_data_summary(sample_df)
    assert summary["shape"] == sample_df.shape
    assert "Age" in summary["numeric_cols"]
    assert "Gender" in summary["categorical_cols"]


def test_summary_counts_nulls_correctly(sample_df) -> None:
    summary = compute_data_summary(sample_df)
    assert summary["total_nulls"] == int(sample_df.isna().sum().sum())
    assert summary["null_counts"]["Score"] == int(sample_df["Score"].isna().sum())


def test_summary_handles_an_all_null_column() -> None:
    import numpy as np
    import pandas as pd

    df = pd.DataFrame({"a": [1, 2, 3], "empty": [np.nan] * 3})
    summary = compute_data_summary(df)
    assert summary["null_counts"]["empty"] == 3
    assert summary["null_pct"]["empty"] == 100.0


def test_summary_handles_single_row() -> None:
    import pandas as pd

    summary = compute_data_summary(pd.DataFrame({"a": [1]}))
    assert summary["shape"] == (1, 1)


def test_summary_handles_mixed_type_column() -> None:
    """`describe(include='all')` can raise on pathological frames."""
    import pandas as pd

    df = pd.DataFrame({"mixed": [1, "two", 3.0, None], "n": [1, 2, 3, 4]})
    summary = compute_data_summary(df)
    assert summary["shape"] == (4, 2)
