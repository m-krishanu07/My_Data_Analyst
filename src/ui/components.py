"""
Reusable render helpers.

Two rules this module exists to enforce:

1. **Escaping.** Several panels use `unsafe_allow_html=True` to get the card
   styling Streamlit's native widgets don't offer. Every value interpolated
   into that HTML is user-controlled (filenames, column names, cell values)
   and is passed through `html.escape` first. A CSV named
   `<img src=x onerror=alert(1)>.csv` previously executed script.

2. **Figure dispatch.** matplotlib figures arrive as PNG bytes, plotly
   figures as JSON. They need different render calls — sending one to the
   other's renderer raises.
"""

from __future__ import annotations

import html
from typing import Any

import pandas as pd
import streamlit as st

from src.logging_conf import get_logger

logger = get_logger(__name__)

_MAX_TABLE_ROWS = 200


def esc(value: Any) -> str:
    """Escape a value for interpolation into raw HTML."""
    return html.escape(str(value), quote=True)


# ── Headers and cards ─────────────────────────────────────────


def page_header(title: str, subtitle: str) -> None:
    """The navy masthead. Emitted as one block so the band wraps both lines."""
    st.markdown(
        f'<div class="masthead">'
        f'<p class="main-title">{esc(title)}</p>'
        f'<p class="sub-title">{esc(subtitle)}</p>'
        f"</div>",
        unsafe_allow_html=True,
    )


def metric_card(label: str, value: Any, note: str | None = None) -> None:
    """A single labelled stat in the sidebar."""
    suffix = f' <span class="muted">{esc(note)}</span>' if note else ""
    st.markdown(
        f'<div class="info-card">'
        f'<div class="info-label">{esc(label)}</div>'
        f'<div class="info-value">{esc(value)}{suffix}</div>'
        f"</div>",
        unsafe_allow_html=True,
    )


def dataset_header(name: str, summary: dict) -> None:
    """Banner above the EDA panels showing the loaded dataset at a glance."""
    rows, cols = summary["shape"]
    stats = [
        ("Rows", f"{rows:,}", ""),
        ("Columns", f"{cols}", ""),
        ("Numeric", len(summary["numeric_cols"]), "accent"),
        ("Categorical", len(summary["categorical_cols"]), "accent"),
        ("Missing", f"{summary['total_nulls']:,}", "danger" if summary["total_nulls"] else ""),
        ("Memory", f"{summary['memory_mb']} MB", ""),
    ]
    cells = "".join(
        f'<div><span class="stat-label">{esc(label)}</span><br>'
        f'<span class="stat-value {cls}">{esc(value)}</span></div>'
        for label, value, cls in stats
    )
    st.markdown(
        f'<div class="dataset-banner">'
        f'<span class="name">{esc(name)}</span>'
        f'<div class="stats">{cells}</div>'
        f"</div>",
        unsafe_allow_html=True,
    )


def empty_state(examples: list[str]) -> None:
    """Landing panel shown before a dataset is loaded."""
    items = "".join(f'<div class="example">{esc(text)}</div>' for text in examples)
    st.markdown(
        '<div class="empty-state">'
        "<h3>No dataset loaded</h3>"
        "<p>Upload a CSV from the sidebar, or load one of the sample datasets "
        "to get started.</p>"
        '<div class="eyebrow">Questions you can ask</div>'
        f"{items}</div>",
        unsafe_allow_html=True,
    )


# ── Tables ────────────────────────────────────────────────────


def render_dataframe(df: pd.DataFrame | None, max_rows: int = 50) -> None:
    """
    Render a DataFrame as a themed HTML table.

    `to_html(escape=True)` is pandas' default and escapes both cell values
    and column names, so untrusted CSV content cannot inject markup here.
    """
    if df is None or (hasattr(df, "empty") and df.empty):
        st.info("No data to display.")
        return

    limit = min(max_rows, _MAX_TABLE_ROWS)
    try:
        markup = df.head(limit).to_html(
            index=True, classes="styled-table", border=0, escape=True, na_rep="—"
        )
    except Exception:
        logger.exception("Falling back to st.dataframe")
        st.dataframe(df.head(limit), use_container_width=True)
        return

    st.markdown(markup, unsafe_allow_html=True)
    if len(df) > limit:
        st.caption(f"Showing {limit:,} of {len(df):,} rows.")


# ── Figures ───────────────────────────────────────────────────


def render_figure(figure: Any, figure_type: str | None, key: str) -> None:
    """
    Render whichever figure format the sandbox produced.

    The sandbox cannot return live objects across the process boundary, so
    matplotlib arrives as PNG bytes and plotly as JSON. Plotly is rebuilt
    into a real Figure so charts stay interactive — zoom, hover and legend
    toggling all survive the round trip.
    """
    if figure is None:
        return

    if figure_type == "png":
        st.image(figure, use_container_width=True)
        return

    if figure_type == "plotly_json":
        try:
            import plotly.io as pio

            st.plotly_chart(pio.from_json(figure), use_container_width=True, key=key)
        except Exception:
            logger.exception("Could not rebuild plotly figure")
            st.warning("The chart could not be displayed.")
        return

    logger.warning("Unknown figure type %r", figure_type)


# ── Run metadata ──────────────────────────────────────────────


def render_run_details(result, key: str) -> None:
    """Generated code, retry history, and the cost/latency line."""
    if result.generated_code:
        with st.expander("View generated code"):
            st.code(result.generated_code, language="python")

    if result.failed_attempts:
        with st.expander(f"Self-correction ({len(result.failed_attempts)} retry)"):
            st.caption(
                "The first attempt failed. The error was fed back to the model, "
                "which rewrote the code."
            )
            for i, attempt in enumerate(result.failed_attempts, start=1):
                st.markdown(f"**Attempt {i} — failed**")
                st.code(attempt["code"], language="python")
                st.error(attempt["error"])

    usage_footer(result)


def usage_footer(result) -> None:
    """One dim line: model, tokens, latency, cost."""
    from src.llm.pricing import format_cost

    if not result.model:
        return

    parts = [esc(result.model)]
    if result.total_tokens:
        parts.append(f"{result.total_tokens:,} tokens")
    if result.latency_s:
        parts.append(f"{result.latency_s:.1f}s")
    if result.cost_usd:
        parts.append(format_cost(result.cost_usd))

    line = '<span class="sep">·</span>'.join(parts)
    if result.repaired:
        line += '<span class="sep">·</span><span class="repaired">self-corrected</span>'
    if result.provider == "replay":
        # Label it. The numbers above this line were computed live, but the
        # model's reply was not generated just now, and saying so costs nothing.
        line += '<span class="sep">·</span><span class="replayed">reply replayed</span>'

    st.markdown(f'<div class="run-meta">{line}</div>', unsafe_allow_html=True)
