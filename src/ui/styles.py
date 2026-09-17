"""
Stylesheet for the app.

Kept out of `app.py` so the page logic stays readable. Colours are defined
once as CSS custom properties; the palette matches `.streamlit/config.toml`
so Streamlit's own widgets and these rules agree instead of fighting.

Design notes, so later edits don't undo the intent:

* **No gradients, no glows, no hover transforms.** Hierarchy comes from the
  type scale, weight and whitespace. A gradient headline and a lifting,
  glowing button are the two fastest ways back to a generic template.
* **Three typefaces, each with a job.** Serif for headings, sans for UI
  chrome, mono for anything numeric. The mono is load-bearing: it gives
  figures a consistent width so columns of numbers line up.
* **Warm neutrals, never pure black or pure white on the page.** Warm stone
  against `#1a1a17` ink reads as print; `#fff` on `#000` reads as a terminal.
* **Three distinct depth levels.** Page (stone) < sidebar (light) < cards
  (white). An earlier revision set all three within 3% of each other and the
  whole page dissolved into one flat sheet — panels need somewhere to lift
  *from*. Keep the gap between `--paper` and `--surface` wide.
"""

from __future__ import annotations

import streamlit as st

# Single source of truth for the palette. Anything that needs a colour in
# Python (not CSS) should import from here rather than hard-coding a hex.
# `.streamlit/config.toml` and the chart theme in `src/sandbox/worker.py`
# both mirror these values and must be updated alongside them.
PALETTE = {
    "paper": "#e8e4dc",
    "sidebar": "#f3f1ec",
    "surface": "#ffffff",
    "surface_alt": "#f6f4ef",
    "border": "#dcd7cc",
    "border_strong": "#c7c1b4",
    "ink": "#1a1a17",
    "ink_soft": "#4a4a44",
    "muted": "#7d7a72",
    "accent": "#14395e",
    "accent_hover": "#0d2740",
    "accent_wash": "#eef2f7",
    "accent_line": "#c4d2e0",
    "danger": "#9b2c2c",
    "positive": "#2f6b4f",
}


def _grain(opacity: float) -> str:
    """
    A paper grain, as an inline SVG turbulence filter.

    Generated rather than shipped as a PNG: it keeps the repo asset-free and
    costs no extra request. Already percent-encoded, so it can be dropped
    straight into a `url()` — note `%23` for the `#` in the filter reference,
    which browsers will otherwise read as a fragment and silently ignore.
    """
    svg = (
        "%3Csvg xmlns='http://www.w3.org/2000/svg' width='160' height='160'%3E"
        "%3Cfilter id='n'%3E"
        "%3CfeTurbulence type='fractalNoise' baseFrequency='0.8' "
        "numOctaves='4' stitchTiles='stitch'/%3E"
        "%3CfeColorMatrix type='saturate' values='0'/%3E"
        "%3C/filter%3E"
        f"%3Crect width='160' height='160' filter='url(%23n)' opacity='{opacity}'/%3E"
        "%3C/svg%3E"
    )
    return f'url("data:image/svg+xml,{svg}")'


_CSS_TEMPLATE = """
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@400;500;600&family=Source+Serif+4:opsz,wght@8..60,400;8..60,600&display=swap');

:root {
    --paper: #e8e4dc;
    --sidebar: #f3f1ec;
    --surface: #ffffff;
    --surface-alt: #f6f4ef;
    --border: #dcd7cc;
    --border-strong: #c7c1b4;
    --ink: #1a1a17;
    --ink-soft: #4a4a44;
    --muted: #7d7a72;
    --accent: #14395e;
    --accent-hover: #0d2740;
    --accent-wash: #eef2f7;
    --accent-line: #c4d2e0;
    --danger: #9b2c2c;
    --positive: #2f6b4f;

    /* Two stops, both barely there. A card should read as sitting on the
       page, not hovering above it. */
    --lift: 0 1px 2px rgba(26, 26, 23, 0.04), 0 1px 3px rgba(26, 26, 23, 0.05);
    --lift-hi: 0 1px 2px rgba(26, 26, 23, 0.05), 0 4px 12px rgba(26, 26, 23, 0.07);

    --sans: 'IBM Plex Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    --serif: 'Source Serif 4', Georgia, serif;
    --mono: 'IBM Plex Mono', ui-monospace, SFMono-Regular, monospace;
}

html, body, .stApp, [data-testid="stAppViewContainer"] {
    font-family: var(--sans);
    color: var(--ink);
    -webkit-font-smoothing: antialiased;
}
html, body { background: var(--paper); }

/* ── Page background ────────────────────────────────────────
   A single flat fill reads as dead space, so the page carries four
   stacked layers: a fine grain on top, then three very low-opacity
   colour pools drawn from the chart colorway (navy, sienna, green).
   The pools are large and soft enough to register as depth rather
   than as shapes.

   Opacity is the whole game here. Every value below is under 11%,
   because cards and charts sit on opaque white — the background is
   allowed to be interesting only where there is no data on top of
   it. Raising these will start to fight the tables.

   The grain is an inline SVG turbulence filter rather than an image
   file, so this stays dependency-free and adds no asset to load.
   It is layered as a background-image rather than a ::before
   overlay deliberately: a fixed pseudo-element has to be stacked
   above the page but below every Streamlit widget, and that fight
   is not worth having. */
.stApp {
    background-color: var(--paper);
    background-image:
        __GRAIN_PAGE__,
        radial-gradient(900px 520px at 10% -8%, rgba(20, 57, 94, 0.10), transparent 62%),
        radial-gradient(820px 480px at 97% 4%, rgba(156, 66, 33, 0.055), transparent 64%),
        radial-gradient(1200px 760px at 55% 106%, rgba(47, 107, 79, 0.055), transparent 66%);
    background-repeat: repeat, no-repeat, no-repeat, no-repeat;
    background-size: 160px 160px, auto, auto, auto;
    background-attachment: fixed, fixed, fixed, fixed;
}
/* Transparent, or it paints over everything above. */
[data-testid="stAppViewContainer"],
[data-testid="stMain"],
[data-testid="stMainBlockContainer"] { background: transparent; }

/* Numerals line up in columns wherever they appear. */
.mono, .info-value, .stat-value, .run-meta, .styled-table td, .styled-table th {
    font-variant-numeric: tabular-nums;
}

h1, h2, h3, h4, h5, h6 {
    font-family: var(--serif);
    color: var(--ink);
    font-weight: 600;
    letter-spacing: -0.01em;
}

a { color: var(--accent); text-underline-offset: 2px; }
a:hover { color: var(--accent-hover); }

code {
    font-family: var(--mono);
    font-size: 0.85em;
    color: var(--accent);
    background: var(--surface-alt);
    padding: 1px 5px;
    border-radius: 3px;
}

pre {
    background: var(--surface-alt);
    border: 1px solid var(--border);
    border-radius: 4px;
}
pre code { color: var(--ink); background: transparent; padding: 0; }

/* ── Masthead ───────────────────────────────────────────────
   The page's one block of saturated colour, and its focal anchor.
   Flat navy with a faint diagonal hatch — texture rather than a
   gradient, so it reads as printed stock instead of a UI kit. */
.masthead {
    background-color: var(--accent);
    background-image: repeating-linear-gradient(
        135deg,
        rgba(255, 255, 255, 0.035) 0 1px,
        transparent 1px 8px
    );
    border-radius: 4px;
    padding: 30px 34px;
    margin-bottom: 22px;
    box-shadow: var(--lift);
}
.main-title {
    font-family: var(--serif);
    color: #ffffff;
    font-size: 2.3rem;
    font-weight: 600;
    letter-spacing: -0.02em;
    line-height: 1.12;
    margin: 0 0 6px 0;
}
.sub-title {
    color: rgba(255, 255, 255, 0.72);
    font-size: 0.94rem;
    font-weight: 400;
    line-height: 1.5;
    margin: 0;
    max-width: 60ch;
}

/* ── Sidebar stats ──────────────────────────────────────────
   A definition list, not a card. Label above value, hairline
   between entries — the density of a report, not a dashboard. */
.info-card {
    padding: 9px 0;
    border-bottom: 1px solid var(--border);
}
.info-label {
    color: var(--muted);
    font-size: 0.7rem;
    font-weight: 500;
    letter-spacing: 0.04em;
    text-transform: uppercase;
    margin-bottom: 3px;
}
.info-value {
    font-family: var(--mono);
    color: var(--ink);
    font-size: 0.95rem;
    font-weight: 500;
    line-height: 1.3;
}
.info-value .muted {
    font-family: var(--sans);
    font-size: 0.8rem;
    color: var(--muted);
    font-weight: 400;
}

/* ── Dataset header ─────────────────────────────────────── */
.dataset-banner {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 4px;
    padding: 16px 20px;
    margin-bottom: 18px;
    box-shadow: var(--lift);
}
.dataset-banner .name {
    font-family: var(--mono);
    color: var(--ink);
    font-size: 0.9rem;
    font-weight: 500;
    display: block;
    margin-bottom: 14px;
    word-break: break-all;
}
.dataset-banner .stats {
    display: flex;
    flex-wrap: wrap;
    gap: 0;
}
.dataset-banner .stats > div {
    padding: 0 20px;
    border-left: 1px solid var(--border);
}
.dataset-banner .stats > div:first-child {
    padding-left: 0;
    border-left: none;
}
.dataset-banner .stat-label {
    color: var(--muted);
    font-size: 0.68rem;
    font-weight: 500;
    letter-spacing: 0.04em;
    text-transform: uppercase;
}
.dataset-banner .stat-value {
    font-family: var(--mono);
    font-size: 1.05rem;
    font-weight: 500;
    color: var(--ink);
}
.dataset-banner .stat-value.accent { color: var(--accent); }
.dataset-banner .stat-value.danger { color: var(--danger); }

/* ── Empty state ────────────────────────────────────────── */
.empty-state {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 4px;
    box-shadow: var(--lift);
    padding: 34px 38px;
    max-width: 680px;
}
.empty-state h3 {
    font-family: var(--serif);
    font-size: 1.45rem;
    font-weight: 600;
    margin: 0 0 8px 0;
}
.empty-state p {
    color: var(--ink-soft);
    font-size: 0.92rem;
    line-height: 1.65;
    margin: 0 0 22px 0;
}
.empty-state .eyebrow {
    color: var(--muted);
    font-size: 0.68rem;
    font-weight: 500;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    padding-bottom: 8px;
    border-bottom: 1px solid var(--border-strong);
    margin-bottom: 2px;
}
.empty-state .example {
    color: var(--ink-soft);
    font-size: 0.9rem;
    padding: 11px 0;
    border-bottom: 1px solid var(--border);
}

/* ── Tables ─────────────────────────────────────────────── */
.styled-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.85rem;
    font-family: var(--sans);
    margin: 4px 0;
    display: block;
    overflow-x: auto;
}
.styled-table thead th {
    color: var(--accent);
    font-weight: 600;
    font-size: 0.68rem;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    padding: 8px 14px;
    text-align: left;
    background: transparent;
    border-bottom: 1.5px solid var(--accent);
    white-space: nowrap;
}
.styled-table tbody tr { border-bottom: 1px solid var(--border); }
.styled-table tbody tr:nth-child(even) { background: var(--surface-alt); }
.styled-table tbody tr:hover { background: var(--accent-wash); }
.styled-table tbody td {
    color: var(--ink-soft);
    font-family: var(--mono);
    font-size: 0.8rem;
    padding: 7px 14px;
    max-width: 320px;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
}
/* pandas puts the index in <th> inside <tbody>. */
.styled-table tbody th {
    color: var(--muted);
    font-family: var(--mono);
    font-size: 0.78rem;
    font-weight: 400;
    padding: 7px 14px;
    text-align: left;
}

/* ── Streamlit widgets ──────────────────────────────────── */
/* Same grain as the page, so the sidebar reads as the same paper stock
   cut to a different shade rather than a slab dropped on top. */
[data-testid="stSidebar"] {
    background-color: var(--sidebar);
    background-image: __GRAIN_SIDEBAR__;
    background-size: 160px 160px;
    border-right: 1px solid var(--border-strong);
}
/* Navy, not grey. The section markers are the sidebar's only structure. */
[data-testid="stSidebar"] h3 {
    font-family: var(--sans);
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: var(--accent);
    margin-bottom: 4px;
}

[data-testid="stExpander"] {
    border: 1px solid var(--border);
    border-radius: 4px;
    background: var(--surface);
    box-shadow: var(--lift);
    margin-bottom: 10px;
}
[data-testid="stExpander"] summary {
    color: var(--accent);
    font-size: 0.88rem;
    font-weight: 600;
}
[data-testid="stExpander"] summary:hover { color: var(--accent-hover); }
[data-testid="stExpander"] summary svg { fill: var(--accent); }

[data-testid="stChatMessage"] {
    border-radius: 4px;
    border: 1px solid var(--border);
    background: var(--surface);
    box-shadow: var(--lift);
    padding: 16px 18px;
}
/* The testid is `stChatMessageAvatarUser` as of Streamlit 1.50. An earlier
   revision used `chatAvatarIcon-user`, a name from an older release that no
   longer exists in the bundle — so this rule silently never applied and user
   and assistant rows looked identical. Verify against the frontend bundle
   before changing it. */
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
    background: var(--accent-wash);
    border-color: var(--accent-line);
    border-left: 2px solid var(--accent);
}

/* The input sits in a fixed bar at the bottom; without this it floats on a
   white strip that doesn't belong to the page. Translucent rather than
   solid, so the background wash shows through instead of being cut off by
   a flat band across the bottom of the page. */
[data-testid="stBottom"] > div { background: transparent; }
[data-testid="stBottomBlockContainer"] {
    background: rgba(232, 228, 220, 0.82);
    backdrop-filter: blur(10px);
    -webkit-backdrop-filter: blur(10px);
}
[data-testid="stChatInput"] {
    background: var(--surface);
    border: 1px solid var(--border-strong);
    border-radius: 4px;
    box-shadow: var(--lift-hi);
}
[data-testid="stChatInput"] textarea {
    background: transparent;
    color: var(--ink);
    font-family: var(--sans);
}
[data-testid="stChatInput"] textarea::placeholder { color: var(--muted); }
[data-testid="stChatInput"]:focus-within { border-color: var(--accent); }

/* Streamlit's dropzone already draws a bordered box. Styling the outer
   wrapper too produced a box nested inside a box. */
[data-testid="stFileUploader"] {
    background: transparent;
    border: none;
    padding: 0;
}
[data-testid="stFileUploaderDropzone"] {
    background: var(--surface);
    border: 1px dashed var(--border-strong);
    border-radius: 4px;
}
[data-testid="stFileUploader"] button {
    background: var(--surface);
    color: var(--accent);
    border: 1px solid var(--border-strong);
    border-radius: 3px;
}

/* Flat. No lift, no glow — the hover changes colour and nothing else. */
.stButton > button {
    border-radius: 3px;
    border: 1px solid var(--border-strong);
    background: var(--surface);
    color: var(--ink);
    font-family: var(--sans);
    font-size: 0.85rem;
    font-weight: 500;
    transition: background 0.12s ease, border-color 0.12s ease, color 0.12s ease;
}
.stButton > button:hover {
    background: var(--accent-wash);
    border-color: var(--accent);
    color: var(--accent);
}
.stButton > button:focus:not(:active) {
    border-color: var(--accent);
    color: var(--accent);
}

[data-testid="stAlert"] { border-radius: 4px; font-size: 0.88rem; }
[data-testid="stImage"] img { border-radius: 3px; }
[data-testid="stCaptionContainer"], .stCaption { color: var(--muted); }
hr { border-color: var(--border); opacity: 1; }

/* ── Scrollbar ──────────────────────────────────────────── */
::-webkit-scrollbar { width: 9px; height: 9px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: var(--border-strong); border-radius: 5px; }
::-webkit-scrollbar-thumb:hover { background: var(--muted); }

/* ── Run metadata line under an answer ──────────────────── */
.run-meta {
    font-family: var(--mono);
    color: var(--muted);
    font-size: 0.72rem;
    margin-top: 12px;
    padding-top: 10px;
    border-top: 1px solid var(--border);
}
.run-meta .sep { opacity: 0.45; margin: 0 7px; }
.run-meta .repaired { color: var(--accent); }
.run-meta .replayed { font-style: italic; }
</style>
"""

# The grain is lighter on the sidebar: it sits on a paler fill, where the same
# opacity reads as dirt rather than texture.
_CSS = _CSS_TEMPLATE.replace("__GRAIN_PAGE__", _grain(0.28)).replace(
    "__GRAIN_SIDEBAR__", _grain(0.2)
)


def inject_styles() -> None:
    """
    Apply the stylesheet.

    Safe to call on every rerun — Streamlit replaces the element in place
    rather than appending a new `<style>` block each time.
    """
    st.markdown(_CSS, unsafe_allow_html=True)
