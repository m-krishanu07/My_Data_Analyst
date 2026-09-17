"""
My Data Analyst — Streamlit entrypoint.

Ask questions about a CSV in plain English. An LLM writes pandas code, the
code runs in an isolated subprocess sandbox, and the result comes back as
text plus a chart.

This file is deliberately thin: styling lives in `src/ui/styles.py`, render
helpers in `src/ui/components.py`, and everything else behind
`SessionManager` / `generate_and_execute`.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from src.agent.agent import generate_and_execute, summarize_for_history
from src.config import ConfigError, settings
from src.llm.base import LLMProvider
from src.llm.replay_provider import Recording, ReplayProvider, load_recordings
from src.logging_conf import get_logger, setup_logging
from src.session_manager import DataLoadError, SessionManager
from src.ui import (
    dataset_header,
    empty_state,
    inject_styles,
    metric_card,
    page_header,
    render_dataframe,
    render_figure,
    render_run_details,
)

setup_logging()
logger = get_logger(__name__)

st.set_page_config(
    page_title="My Data Analyst",
    # A Material glyph rather than an emoji, to match the monochrome UI.
    page_icon=":material/query_stats:",
    layout="wide",
    initial_sidebar_state="expanded",
)
inject_styles()

DEMO_DIR = Path(__file__).parent / "demo_data"
EXAMPLE_QUESTIONS = [
    '"Which region has the highest average order value?"',
    '"Plot the distribution of customer age"',
    '"Is discount correlated with profit?"',
    '"Train a model to predict whether an order is returned"',
]


# ── Loading ───────────────────────────────────────────────────


def load_upload(file_obj) -> None:
    try:
        df = SessionManager.load_df_from_file(file_obj)
    except DataLoadError as exc:
        st.sidebar.error(str(exc))
        return
    SessionManager.init_session(df, file_obj.name)
    st.rerun()


def load_demo(path: Path) -> None:
    try:
        df = SessionManager.load_df_from_path(str(path))
    except DataLoadError as exc:
        st.sidebar.error(str(exc))
        return
    SessionManager.init_session(df, path.name)
    st.rerun()


# ── Demo mode ─────────────────────────────────────────────────


@st.cache_resource(show_spinner=False)
def demo_recordings() -> list[Recording]:
    """Fixture contents, read once per process rather than per rerun."""
    return load_recordings()


def demo_provider() -> ReplayProvider:
    """Offline provider scoped to the dataset currently loaded."""
    name = SessionManager.get_dataset_name()
    return ReplayProvider(
        dataset=Path(name).stem if name else None,
        recordings=demo_recordings(),
    )


def render_demo_notice(provider: ReplayProvider) -> None:
    """Explain what is and isn't real, then offer the answerable questions."""
    questions = provider.available_questions()

    if not questions:
        st.warning(
            "**Demo mode** — no API key is configured, so this build can only "
            "replay recorded answers for the two sample datasets. The preview "
            "and statistics above were computed live from *your* file; to ask "
            "questions about it, set `GROQ_API_KEY` (free, no card, at "
            "console.groq.com/keys) and reload."
        )
        return

    st.info(
        "**Demo mode** — no API key is configured, so the model's replies are "
        "replayed from a recorded benchmark run. Only that network call is "
        "replayed: the code is still validated and executed in the sandbox, so "
        "every number and chart below is computed from this CSV right now."
    )
    st.caption("Questions this demo can answer:")
    for start in range(0, len(questions), 2):
        row = st.columns(2)
        for offset, question in enumerate(questions[start : start + 2]):
            if row[offset].button(
                question, key=f"chip_{start + offset}", use_container_width=True
            ):
                st.session_state["_pending_query"] = question


# ── Sidebar ───────────────────────────────────────────────────


def render_sidebar(demo_mode: bool) -> None:
    with st.sidebar:
        st.markdown("### Dataset")

        upload = st.file_uploader("Upload a CSV", type=["csv"], key="uploader")
        if upload is not None and upload.name != st.session_state.get("_loaded_name"):
            # Guard on the name so a rerun (every keystroke in the chat box)
            # does not re-parse the same file.
            st.session_state["_loaded_name"] = upload.name
            load_upload(upload)

        demos = sorted(DEMO_DIR.glob("*.csv")) if DEMO_DIR.exists() else []
        if demos:
            st.caption("…or try a sample dataset")
            for path in demos:
                label = path.stem.replace("_", " ").title()
                if st.button(label, use_container_width=True, key=f"demo_{path.stem}"):
                    st.session_state["_loaded_name"] = path.name
                    load_demo(path)

        if SessionManager.has_session():
            summary = SessionManager.get_data_summary(SessionManager.get_df())
            rows, cols = summary["shape"]

            st.markdown("---")
            st.markdown("### Overview")
            metric_card("Rows × Columns", f"{rows:,} × {cols}")
            metric_card("Memory", f"{summary['memory_mb']} MB")
            metric_card(
                "Column Types",
                f"{len(summary['numeric_cols'])} numeric · "
                f"{len(summary['categorical_cols'])} categorical",
            )
            metric_card(
                "Missing Values",
                f"{summary['total_nulls']:,}",
                note=f"({summary['null_pct_overall']}%)",
            )

            st.markdown("---")
            if st.button("Clear chat history", use_container_width=True):
                SessionManager.clear_chat()
                st.rerun()

        st.markdown("---")
        st.markdown("### Model")
        if demo_mode:
            st.caption("**Provider** · offline demo — replies replayed")
            st.caption("**Analysis** · executed live in the sandbox")
        else:
            st.caption(f"**Provider** · {settings.provider}")
            st.caption(f"**Model** · {settings.model_for()}")
        st.caption(
            f"**Sandbox** · isolated subprocess, {settings.sandbox.timeout_seconds}s timeout"
        )


# ── Exploratory panels ────────────────────────────────────────


def render_eda(summary: dict) -> None:
    with st.expander("Data preview", expanded=True):
        render_dataframe(summary["sample"], max_rows=settings.data.preview_rows)

    with st.expander("Statistical summary"):
        render_dataframe(summary["describe"], max_rows=50)

    with st.expander("Missing values"):
        missing = [
            {
                "Column": col,
                "Nulls": summary["null_counts"][col],
                "Null %": summary["null_pct"][col],
            }
            for col in summary["columns"]
            if summary["null_counts"][col] > 0
        ]
        if missing:
            frame = (
                pd.DataFrame(missing).sort_values("Nulls", ascending=False).reset_index(drop=True)
            )
            render_dataframe(frame, max_rows=100)
        else:
            st.success("No missing values in this dataset.")

    with st.expander("Column details"):
        details = pd.DataFrame(
            [
                {
                    "Column": col,
                    "Type": summary["dtypes"][col],
                    "Unique": summary["unique_counts"][col],
                    "Missing": summary["null_counts"][col],
                }
                for col in summary["columns"]
            ]
        )
        render_dataframe(details, max_rows=200)


# ── Chat ──────────────────────────────────────────────────────


def replay_history() -> None:
    for i, msg in enumerate(SessionManager.get_chat_messages()):
        with st.chat_message(msg["role"]):
            if msg.get("is_error"):
                st.error(msg["content"])
            else:
                st.markdown(msg["content"])
            render_figure(msg.get("figure"), msg.get("figure_type"), key=f"hist_fig_{i}")
            if msg.get("code"):
                with st.expander("View generated code"):
                    st.code(msg["code"], language="python")


def answer(query: str, df: pd.DataFrame, provider: LLMProvider | None = None) -> None:
    with st.chat_message("assistant"):
        with st.spinner("Analysing…"):
            result = generate_and_execute(
                query,
                df,
                conversation_history=SessionManager.get_conversation_history(),
                provider=provider,
            )

        SessionManager.add_to_history("user", query)

        if not result.ok:
            # The user sees a clean message; the traceback stays in the logs
            # and never enters the model's context.
            st.error(result.error)
            SessionManager.add_chat_message("assistant", result.error, is_error=True)
            SessionManager.add_to_history("assistant", f"(failed: {result.error[:200]})")
            if result.generated_code:
                with st.expander("View generated code"):
                    st.code(result.generated_code, language="python")
            return

        blocks = []
        if result.explanation:
            st.markdown(result.explanation)
            blocks.append(result.explanation)

        if result.result_text:
            st.markdown(result.result_text)
            blocks.append(result.result_text)

        render_figure(result.figure, result.figure_type, key="live_fig")

        if result.stdout:
            with st.expander("Printed output"):
                st.code(result.stdout)

        if result.warning:
            st.warning(result.warning)

        render_run_details(result, key="live")

        content = "\n\n".join(blocks) or "(no output)"
        SessionManager.add_chat_message(
            "assistant",
            content,
            figure=result.figure,
            figure_type=result.figure_type,
            code=result.generated_code,
        )
        SessionManager.add_to_history("assistant", summarize_for_history(result))


# ── Page ──────────────────────────────────────────────────────


def main() -> None:
    page_header(
        "My Data Analyst",
        "Ask questions about a CSV in plain English. Answers and charts are computed live.",
    )

    # No credentials is not a fatal error: the app drops to replaying recorded
    # answers so a deployed link keeps working after a key rotates or a
    # free-tier quota trips. A *misconfigured* provider still is fatal — that
    # is a mistake worth surfacing, not papering over.
    demo_mode = settings.demo_mode()
    if not demo_mode:
        try:
            settings.validate_provider()
        except ConfigError as exc:
            st.error(f"**Configuration problem**\n\n```\n{exc}\n```")
            st.stop()

    render_sidebar(demo_mode)

    if not SessionManager.has_session():
        if demo_mode:
            st.info(
                "**Demo mode** — no API key is configured. Load a sample dataset "
                "from the sidebar to see recorded answers executed live, or set "
                "`GROQ_API_KEY` to ask anything about your own CSV."
            )
        empty_state(EXAMPLE_QUESTIONS)
        st.stop()

    df = SessionManager.get_df()
    summary = SessionManager.get_data_summary(df)

    dataset_header(SessionManager.get_dataset_name(), summary)
    render_eda(summary)
    st.markdown("---")

    provider = None
    if demo_mode:
        provider = demo_provider()
        render_demo_notice(provider)

    replay_history()

    # A question chip sets `_pending_query` earlier in this same run, so it is
    # read here rather than requiring a second rerun.
    query = st.session_state.pop("_pending_query", None) or st.chat_input(
        "Ask anything about your data…"
    )
    if query:
        SessionManager.add_chat_message("user", query)
        with st.chat_message("user"):
            st.markdown(query)
        answer(query, df, provider)


main()
