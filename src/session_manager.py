"""
Session state and CSV ingestion.

Streamlit re-runs the whole script on every interaction, so anything
expensive here runs on every keystroke unless it is cached, and any
unhandled exception takes down the entire page.
"""

from __future__ import annotations

import csv
import io

import pandas as pd
import streamlit as st

from src.config import settings
from src.logging_conf import get_logger

logger = get_logger(__name__)

_ENCODINGS = ("utf-8", "utf-8-sig", "latin-1", "cp1252")


class DataLoadError(ValueError):
    """Raised when an upload cannot be turned into a usable DataFrame."""


def _sniff_delimiter(sample: str) -> str:
    """
    Detect the delimiter so semicolon- and tab-separated exports work.

    Excel in many locales writes `;` separated files with a .csv extension;
    without this they load as a single column and every question fails.
    """
    try:
        return csv.Sniffer().sniff(sample, delimiters=",;\t|").delimiter
    except csv.Error:
        return ","


def _is_control(char: str) -> bool:
    """Unprintable characters, excluding the whitespace CSVs legitimately use."""
    codepoint = ord(char)
    if char in "\t\n\r":
        return False
    return codepoint < 0x20 or 0x7F <= codepoint <= 0x9F


def _looks_binary(text: str) -> bool:
    """
    Detect non-text uploads.

    latin-1 decodes any byte sequence without error, so the encoding
    fallback can never reject a binary file on its own. Without this check
    a PNG renamed to .csv parses into a meaningless frame of `Unnamed: 0`
    columns and every subsequent question fails for no visible reason.

    The test runs on decoded text rather than raw bytes so that multi-byte
    UTF-8 accents (`José`, `Köln`) are not mistaken for binary — checking
    raw bytes rejected legitimate non-English CSVs.
    """
    sample = text[:8192]
    if not sample:
        return False
    if "\x00" in sample:
        return True
    control = sum(1 for char in sample if _is_control(char))
    return control / len(sample) > 0.05


def _decode(raw: bytes) -> str:
    for encoding in _ENCODINGS:
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    # Last resort: never fail the upload purely over a few bad bytes.
    return raw.decode("utf-8", errors="replace")


def _dedupe_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Make column names unique.

    Duplicates make `df['col']` return a DataFrame instead of a Series,
    which breaks generated code in confusing ways.
    """
    if not df.columns.duplicated().any():
        return df
    seen: dict[str, int] = {}
    renamed = []
    for col in df.columns:
        if col in seen:
            seen[col] += 1
            renamed.append(f"{col}.{seen[col]}")
        else:
            seen[col] = 0
            renamed.append(col)
    logger.info("Renamed duplicate columns in upload")
    df.columns = renamed
    return df


def parse_csv_bytes(raw: bytes, filename: str = "uploaded file") -> pd.DataFrame:
    """Parse raw CSV bytes into a DataFrame, raising DataLoadError on failure."""
    if not raw:
        raise DataLoadError(f"{filename} is empty.")

    size_mb = len(raw) / 1024**2
    if size_mb > settings.data.max_upload_mb:
        raise DataLoadError(
            f"{filename} is {size_mb:.1f} MB, which exceeds the "
            f"{settings.data.max_upload_mb} MB limit."
        )

    text = _decode(raw)
    if not text.strip():
        raise DataLoadError(f"{filename} contains no data.")

    if _looks_binary(text):
        raise DataLoadError(
            f"{filename} does not look like a text CSV file. "
            "Export your data as CSV and try again."
        )

    delimiter = _sniff_delimiter(text[:8192])

    try:
        df = pd.read_csv(
            io.StringIO(text),
            sep=delimiter,
            nrows=settings.data.max_rows,
            skip_blank_lines=True,
        )
    except pd.errors.EmptyDataError as exc:
        raise DataLoadError(f"{filename} has no columns to parse.") from exc
    except pd.errors.ParserError as exc:
        raise DataLoadError(
            f"Could not parse {filename} as CSV — the rows may have inconsistent "
            f"column counts. Details: {exc}"
        ) from exc
    except Exception as exc:
        raise DataLoadError(f"Could not read {filename}: {exc}") from exc

    if df.empty:
        raise DataLoadError(f"{filename} parsed successfully but contains no rows.")
    if len(df.columns) == 0:
        raise DataLoadError(f"{filename} contains no columns.")

    df.columns = [str(c).strip() for c in df.columns]
    return _dedupe_columns(df)


@st.cache_data(show_spinner=False, max_entries=8)
def compute_data_summary(df: pd.DataFrame) -> dict:
    """
    Descriptive statistics for the overview panels.

    Cached because `describe(include='all')` on a large frame is expensive
    and Streamlit would otherwise recompute it on every rerun — including
    every keystroke in the chat box.
    """
    numeric_cols = list(df.select_dtypes(include="number").columns)
    categorical_cols = list(df.select_dtypes(include=["object", "category", "bool"]).columns)
    datetime_cols = list(df.select_dtypes(include=["datetime", "datetimetz"]).columns)

    try:
        described = df.describe(include="all").round(3)
    except (ValueError, TypeError):
        described = df.describe().round(3)

    null_counts = df.isna().sum()
    total_cells = int(df.shape[0]) * int(df.shape[1])

    return {
        "shape": df.shape,
        "columns": list(df.columns),
        "dtypes": df.dtypes.astype(str).to_dict(),
        "describe": described,
        "null_counts": null_counts.to_dict(),
        "null_pct": (df.isna().mean() * 100).round(2).to_dict(),
        "total_nulls": int(null_counts.sum()),
        "null_pct_overall": round(float(null_counts.sum()) / total_cells * 100, 2)
        if total_cells
        else 0.0,
        "unique_counts": df.nunique(dropna=True).to_dict(),
        "sample": df.head(settings.data.preview_rows),
        "memory_mb": round(df.memory_usage(deep=True).sum() / 1024**2, 2),
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
        "datetime_cols": datetime_cols,
        "duplicate_rows": int(df.duplicated().sum()),
    }


class SessionManager:
    """Namespaced accessors for Streamlit session state."""

    DF_KEY = "conversational_df"
    HISTORY_KEY = "conversation_history"
    MESSAGES_KEY = "chat_messages"
    NAME_KEY = "dataset_name"

    # ── Dataset ───────────────────────────────────────────────

    @staticmethod
    def init_session(df: pd.DataFrame, name: str = "dataset.csv") -> None:
        st.session_state[SessionManager.DF_KEY] = df
        st.session_state[SessionManager.NAME_KEY] = name
        # A new dataset invalidates prior context — old column references
        # would mislead the model.
        st.session_state[SessionManager.HISTORY_KEY] = []
        st.session_state[SessionManager.MESSAGES_KEY] = []
        logger.info("Loaded '%s' (%d rows x %d cols)", name, len(df), len(df.columns))

    @staticmethod
    def has_session() -> bool:
        return st.session_state.get(SessionManager.DF_KEY) is not None

    @staticmethod
    def get_df() -> pd.DataFrame | None:
        return st.session_state.get(SessionManager.DF_KEY)

    @staticmethod
    def get_dataset_name() -> str:
        return st.session_state.get(SessionManager.NAME_KEY, "dataset.csv")

    @staticmethod
    def clear_dataset() -> None:
        for key in (
            SessionManager.DF_KEY,
            SessionManager.NAME_KEY,
            SessionManager.HISTORY_KEY,
            SessionManager.MESSAGES_KEY,
        ):
            st.session_state.pop(key, None)

    # ── Loading ───────────────────────────────────────────────

    @staticmethod
    def load_df_from_file(file_obj) -> pd.DataFrame:
        name = getattr(file_obj, "name", "uploaded file")
        return parse_csv_bytes(file_obj.getvalue(), name)

    @staticmethod
    def load_df_from_path(path: str) -> pd.DataFrame:
        from pathlib import Path

        file_path = Path(path)
        if not file_path.exists():
            raise DataLoadError(f"File not found: {path}")
        return parse_csv_bytes(file_path.read_bytes(), file_path.name)

    # ── Conversation history (model-facing) ───────────────────

    @staticmethod
    def get_conversation_history() -> list[dict]:
        return st.session_state.get(SessionManager.HISTORY_KEY, [])

    @staticmethod
    def add_to_history(role: str, content: str) -> None:
        history = st.session_state.setdefault(SessionManager.HISTORY_KEY, [])
        history.append({"role": role, "content": content})
        # Two entries per turn; trim to bound prompt size and cost.
        limit = settings.agent.history_turns * 2
        if len(history) > limit:
            st.session_state[SessionManager.HISTORY_KEY] = history[-limit:]

    # ── Chat messages (UI-facing) ─────────────────────────────

    @staticmethod
    def get_chat_messages() -> list[dict]:
        return st.session_state.get(SessionManager.MESSAGES_KEY, [])

    @staticmethod
    def add_chat_message(role: str, content: str, **extra) -> None:
        messages = st.session_state.setdefault(SessionManager.MESSAGES_KEY, [])
        messages.append({"role": role, "content": content, **extra})

    @staticmethod
    def clear_chat() -> None:
        st.session_state[SessionManager.HISTORY_KEY] = []
        st.session_state[SessionManager.MESSAGES_KEY] = []

    # ── Summary ───────────────────────────────────────────────

    @staticmethod
    def get_data_summary(df: pd.DataFrame) -> dict:
        return compute_data_summary(df)
