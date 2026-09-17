"""
UI safety.

S5: the uploaded filename was interpolated straight into a
`st.markdown(..., unsafe_allow_html=True)` block, so a file named
`<img src=x onerror=...>.csv` executed script in every visitor's browser.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.ui.components import esc

XSS_PAYLOADS = [
    "<script>alert(1)</script>",
    '<img src=x onerror="alert(1)">',
    '"><svg onload=alert(1)>',
    "<iframe src=javascript:alert(1)>",
    "</div><script>fetch('//evil')</script>",
    "' onmouseover='alert(1)",
]


@pytest.mark.parametrize("payload", XSS_PAYLOADS)
def test_escape_neutralises_markup(payload: str) -> None:
    escaped = esc(payload)
    assert "<" not in escaped
    assert ">" not in escaped
    assert '"' not in escaped


@pytest.mark.parametrize("payload", XSS_PAYLOADS)
def test_dataset_header_escapes_the_filename(payload: str) -> None:
    """The exact defect: a malicious filename must not reach the DOM as markup."""
    rendered = _render_dataset_header(f"{payload}.csv")
    assert "<script" not in rendered.lower()
    assert "onerror" not in rendered.lower() or "&lt;" in rendered
    assert "&lt;" in rendered or "&quot;" in rendered or "&#x27;" in rendered


def _render_dataset_header(name: str) -> str:
    """Capture the markup `dataset_header` would emit, without a Streamlit runtime."""
    from unittest.mock import patch

    from src.ui import components

    captured: list[str] = []
    summary = {
        "shape": (10, 3),
        "numeric_cols": ["a"],
        "categorical_cols": ["b"],
        "total_nulls": 0,
        "memory_mb": 0.1,
    }
    with patch.object(components.st, "markdown", lambda html, **kw: captured.append(html)):
        components.dataset_header(name, summary)
    return "".join(captured)


def test_escapes_column_names_from_untrusted_csv() -> None:
    """Column names come from the uploaded file and are equally untrusted."""
    df = pd.DataFrame({"<script>alert(1)</script>": [1, 2]})
    markup = df.to_html(index=True, classes="styled-table", border=0, escape=True)
    assert "<script>" not in markup
    assert "&lt;script&gt;" in markup


def test_escapes_cell_values_from_untrusted_csv() -> None:
    df = pd.DataFrame({"note": ['<img src=x onerror="alert(1)">']})
    markup = df.to_html(index=True, classes="styled-table", border=0, escape=True)
    assert "<img" not in markup
    assert "onerror=&quot;" in markup or "&lt;img" in markup


def test_esc_handles_non_string_values() -> None:
    assert esc(42) == "42"
    assert esc(None) == "None"
    assert esc(3.14) == "3.14"


# ── C1: figure dispatch ───────────────────────────────────────


def test_png_figures_go_to_st_image() -> None:
    """Sending plotly JSON to `st.image` was the original crash."""
    from unittest.mock import MagicMock, patch

    from src.ui import components

    with patch.object(components, "st", MagicMock()) as mock_st:
        components.render_figure(b"\x89PNG\r\n\x1a\nfake", "png", key="k")
        mock_st.image.assert_called_once()
        mock_st.plotly_chart.assert_not_called()


def test_plotly_figures_go_to_plotly_chart() -> None:
    from unittest.mock import MagicMock, patch

    import plotly.express as px

    from src.ui import components

    spec = px.scatter(x=[1, 2, 3], y=[1, 4, 9]).to_json()
    with patch.object(components, "st", MagicMock()) as mock_st:
        components.render_figure(spec, "plotly_json", key="k")
        mock_st.plotly_chart.assert_called_once()
        mock_st.image.assert_not_called()


def test_no_figure_renders_nothing() -> None:
    from unittest.mock import MagicMock, patch

    from src.ui import components

    with patch.object(components, "st", MagicMock()) as mock_st:
        components.render_figure(None, None, key="k")
        mock_st.image.assert_not_called()
        mock_st.plotly_chart.assert_not_called()


def test_corrupt_plotly_json_warns_instead_of_crashing() -> None:
    from unittest.mock import MagicMock, patch

    from src.ui import components

    with patch.object(components, "st", MagicMock()) as mock_st:
        components.render_figure("{not valid json", "plotly_json", key="k")
        mock_st.warning.assert_called_once()
