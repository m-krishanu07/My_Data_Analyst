"""
Sandbox worker process.

Runs as `python -m src.sandbox.worker`. It is a long-lived process that
imports the heavy scientific stack once and then serves execution jobs over
stdin/stdout, so users don't pay a multi-second import cost per query.

Isolation guarantees come from being a *separate OS process*:
  - the parent can hard-kill it, which is the only way to reliably stop
    `while True: pass` (a thread cannot be killed in Python)
  - a segfault or memory blow-up takes down the worker, not the web app
  - on POSIX an address-space rlimit is applied

The restricted builtins/imports below are defence in depth. The AST
validator (validator.py) is the primary gate and runs in the parent.
"""

from __future__ import annotations

import io
import os
import sys

# ── Claim the real stdout before anything else can write to it ────────
# stdout is the protocol channel. If a library banner or a user `print()`
# leaked into it, the length-prefixed stream would desynchronise and every
# subsequent response would be garbage. So we duplicate the pipe to a
# private fd, then point fd 1 at stderr.
_PROTOCOL_OUT = os.fdopen(os.dup(sys.stdout.fileno()), "wb", buffering=0)
os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
_PROTOCOL_IN = sys.stdin.buffer

import builtins
import contextlib
import traceback

import matplotlib

matplotlib.use("Agg")  # headless backend — must precede pyplot import

import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import scipy.stats as scipy_stats
import seaborn as sns
import sklearn

from src.sandbox.protocol import read_job, write_result

# ── Chart theme ───────────────────────────────────────────────────────
#
# Set here rather than in the prompt, because a house style the model has to
# remember is a house style it will eventually forget. Defaults apply to any
# figure generated code creates, so charts match the page whatever the model
# writes.
#
# Colours mirror PALETTE in src/ui/styles.py.

_INK = "#1a1a17"
_MUTED = "#85827a"
_GRID = "#ebe8e1"
_SURFACE = "#ffffff"

# Desaturated and print-like, accent navy leading. Ordered so the first three
# stay distinguishable in greyscale and to the most common colour-blindness.
_COLORWAY = ["#14395e", "#9c4221", "#2f6b4f", "#8a6d1f", "#5c4b8a", "#7a7a72"]


def _apply_chart_theme() -> None:
    # seaborn's set_theme rewrites rcParams wholesale, so it has to run first
    # or it silently discards everything set below it.
    sns.set_theme(style="whitegrid", palette=_COLORWAY)

    matplotlib.rcParams.update(
        {
            # matplotlib can only use fonts installed on the machine, so the
            # UI webfont is unavailable here. DejaVu Sans ships with
            # matplotlib and is the guaranteed fallback.
            "font.family": "sans-serif",
            "font.sans-serif": ["IBM Plex Sans", "Segoe UI", "Helvetica", "DejaVu Sans"],
            "font.size": 10,
            # White, not paper — the figure sits inside a white chat row, and
            # a mismatched facecolor shows up as a visible rectangle.
            "figure.facecolor": _SURFACE,
            "axes.facecolor": _SURFACE,
            "savefig.facecolor": _SURFACE,
            "axes.edgecolor": "#d3cfc4",
            "axes.linewidth": 0.8,
            "axes.labelcolor": _INK,
            "axes.labelsize": 10,
            "axes.titlecolor": _INK,
            "axes.titlesize": 12,
            "axes.titleweight": "semibold",
            "axes.titlelocation": "left",
            "axes.titlepad": 12,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.prop_cycle": matplotlib.cycler(color=_COLORWAY),
            # Horizontal rules only: vertical gridlines on a bar chart are
            # noise, and this is the reading direction for a value axis.
            "axes.grid": True,
            "axes.grid.axis": "y",
            "grid.color": _GRID,
            "grid.linewidth": 0.7,
            "text.color": _INK,
            "xtick.color": _MUTED,
            "ytick.color": _MUTED,
            "xtick.labelcolor": _INK,
            "ytick.labelcolor": _INK,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.frameon": False,
            "legend.fontsize": 9,
        }
    )

    pio.templates["analyst"] = go.layout.Template(
        layout=go.Layout(
            colorway=_COLORWAY,
            paper_bgcolor=_SURFACE,
            plot_bgcolor=_SURFACE,
            font={"family": "IBM Plex Sans, Segoe UI, Helvetica, sans-serif",
                  "size": 12, "color": _INK},
            title={"x": 0, "xanchor": "left", "font": {"size": 15}},
            xaxis={"gridcolor": _GRID, "linecolor": "#d3cfc4", "zerolinecolor": _GRID,
                   "ticks": "outside", "tickcolor": _GRID},
            yaxis={"gridcolor": _GRID, "linecolor": "#d3cfc4", "zerolinecolor": _GRID,
                   "ticks": "outside", "tickcolor": _GRID},
            legend={"bgcolor": "rgba(0,0,0,0)"},
            margin={"t": 56, "r": 24, "b": 48, "l": 60},
        )
    )
    pio.templates.default = "analyst"


_apply_chart_theme()

# ── Resource limits (POSIX only) ──────────────────────────────────────


def _apply_memory_limit(limit_mb: int) -> bool:
    """
    Cap the worker's address space. Returns True if the limit was applied.

    Windows has no `resource` module and no equivalent that doesn't pull in
    pywin32 Job Objects, so on Windows this is a no-op and we rely on the
    parent's wall-clock timeout instead. The README states this plainly
    rather than implying a guarantee we don't provide.
    """
    try:
        import resource
    except ImportError:
        return False

    try:
        limit_bytes = limit_mb * 1024 * 1024
        _soft, hard = resource.getrlimit(resource.RLIMIT_AS)
        new_hard = hard if hard != resource.RLIM_INFINITY else limit_bytes
        resource.setrlimit(resource.RLIMIT_AS, (limit_bytes, new_hard))
        return True
    except (ValueError, OSError):
        return False


# ── Restricted execution namespace ────────────────────────────────────

SAFE_BUILTIN_NAMES = frozenset(
    {
        "abs", "all", "any", "bin", "bool", "bytes", "callable", "chr",
        "complex", "dict", "divmod", "enumerate", "filter", "float",
        "format", "frozenset", "hasattr", "hash", "hex", "int", "isinstance",
        "issubclass", "iter", "len", "list", "map", "max", "min", "next",
        "oct", "ord", "pow", "print", "range", "repr", "reversed", "round",
        "set", "slice", "sorted", "str", "sum", "tuple", "type", "zip",
        # exception types — generated code legitimately raises/catches these
        "Exception", "ValueError", "TypeError", "KeyError", "IndexError",
        "ZeroDivisionError", "AttributeError", "RuntimeError",
        "StopIteration", "ArithmeticError", "NotImplementedError",
        "True", "False", "None",
    }
)

_ALLOWED_IMPORT_ROOTS = frozenset(
    {
        "pandas", "numpy", "matplotlib", "seaborn", "plotly", "scipy",
        "sklearn", "statsmodels", "math", "statistics", "random", "re",
        "json", "collections", "itertools", "functools", "datetime",
        "decimal", "fractions", "string", "textwrap", "warnings", "typing",
    }
)

_real_import = builtins.__import__


def _restricted_import(name, globals=None, locals=None, fromlist=(), level=0):
    if level != 0:
        raise ImportError("Relative imports are not allowed in the sandbox.")
    if name.split(".")[0] not in _ALLOWED_IMPORT_ROOTS:
        raise ImportError(f"Import of '{name}' is not allowed in the sandbox.")
    return _real_import(name, globals, locals, fromlist, level)


def _build_namespace() -> dict:
    safe_builtins = {
        name: getattr(builtins, name)
        for name in SAFE_BUILTIN_NAMES
        if hasattr(builtins, name)
    }
    safe_builtins["__import__"] = _restricted_import

    return {
        "__builtins__": safe_builtins,
        "pd": pd,
        "np": np,
        "plt": plt,
        "sns": sns,
        "px": px,
        "go": go,
        "scipy_stats": scipy_stats,
        "sklearn": sklearn,
    }


# ── Result normalisation ──────────────────────────────────────────────


def _is_plotly_figure(obj) -> bool:
    return isinstance(obj, go.Figure) or (
        hasattr(obj, "to_plotly_json") and hasattr(obj, "data")
    )


def _unpack_result(raw):
    """
    Coerce whatever `run()` returned into (result_text, figure).

    Models frequently return a bare value or a bare figure instead of the
    documented 2-tuple. The old code did a blind tuple-unpack and raised an
    opaque ValueError. Being permissive here converts a hard failure into a
    correct answer.
    """
    if isinstance(raw, tuple) and len(raw) == 2:
        return raw[0], raw[1]
    if isinstance(raw, matplotlib.figure.Figure) or _is_plotly_figure(raw):
        return None, raw
    return raw, None


def _serialize_figure(fig, max_bytes: int):
    """Return (figure_payload, figure_type, warning)."""
    if fig is None:
        return None, None, None

    if isinstance(fig, matplotlib.figure.Figure):
        buf = io.BytesIO()
        try:
            fig.savefig(buf, format="png", bbox_inches="tight", dpi=140)
        finally:
            plt.close(fig)
        data = buf.getvalue()
        if len(data) > max_bytes:
            return None, None, "Chart was too large to display."
        return data, "png", None

    if _is_plotly_figure(fig):
        # Ship the figure spec as JSON rather than rasterising it. This
        # renders as a fully interactive chart in Streamlit and removes the
        # hard dependency on kaleido, whose absence previously produced a
        # broken image.
        try:
            spec = pio.to_json(fig)
        except Exception as exc:  # noqa: BLE001 - a bad chart must not kill the job
            return None, None, f"Could not serialize chart: {exc}"
        if len(spec) > max_bytes:
            return None, None, "Chart was too large to display."
        return spec, "plotly_json", None

    return None, None, f"Unsupported figure type: {type(fig).__name__}"


def _stringify(value, max_chars: int) -> str | None:
    if value is None:
        return None
    if isinstance(value, pd.DataFrame):
        text = value.to_markdown(index=False)
    elif isinstance(value, pd.Series):
        text = value.to_frame().to_markdown()
    elif isinstance(value, str):
        text = value
    else:
        text = str(value)
    if len(text) > max_chars:
        text = text[:max_chars] + f"\n\n… output truncated at {max_chars:,} characters."
    return text


def _format_exception(exc: BaseException) -> str:
    """
    Produce a short, model-actionable error string.

    Full tracebacks leak sandbox internals and, when fed back into the
    self-correction loop, waste tokens on frames the model cannot act on.
    We keep only the exception type, message, and the failing line of
    *generated* code.
    """
    tb = exc.__traceback__
    lineno = None
    while tb is not None:
        if tb.tb_frame.f_code.co_filename == "<generated>":
            lineno = tb.tb_lineno
        tb = tb.tb_next

    label = type(exc).__name__
    message = str(exc) or "(no message)"
    if lineno is not None:
        return f"{label} on line {lineno} of the generated code: {message}"
    return f"{label}: {message}"


# ── Job execution ─────────────────────────────────────────────────────


def _execute(job: dict) -> dict:
    code = job["code"]
    df = job["df"]
    max_output_chars = job.get("max_output_chars", 20_000)
    max_figure_bytes = job.get("max_figure_bytes", 8 * 1024 * 1024)

    stdout_capture = io.StringIO()
    namespace = _build_namespace()

    try:
        compiled = compile(code, "<generated>", "exec")
        with contextlib.redirect_stdout(stdout_capture):
            exec(compiled, namespace)  # noqa: S102 - validated + isolated

            run_fn = namespace.get("run")
            if not callable(run_fn):
                return {"error": "Generated code does not define a `run(df)` function."}

            raw = run_fn(df.copy())

        result_value, fig = _unpack_result(raw)
        figure, figure_type, fig_warning = _serialize_figure(fig, max_figure_bytes)

        return {
            "result_text": _stringify(result_value, max_output_chars),
            "figure": figure,
            "figure_type": figure_type,
            "stdout": _stringify(stdout_capture.getvalue().strip() or None, 4_000),
            "warning": fig_warning,
            "error": None,
        }

    except BaseException as exc:  # noqa: BLE001 - must not kill the worker
        return {"error": _format_exception(exc)}
    finally:
        plt.close("all")  # figures accumulate across jobs otherwise


def main() -> None:
    limit_mb = int(os.environ.get("SANDBOX_MEMORY_MB", "2048"))
    memory_limited = _apply_memory_limit(limit_mb)

    write_result(_PROTOCOL_OUT, {"ready": True, "memory_limited": memory_limited})

    while True:
        try:
            job = read_job(_PROTOCOL_IN)
        except Exception:  # noqa: BLE001 - parent closed the pipe; exit quietly
            break
        if job is None or job.get("shutdown"):
            break

        try:
            response = _execute(job)
        except BaseException:  # noqa: BLE001
            response = {"error": f"Sandbox worker failure:\n{traceback.format_exc(limit=3)}"}

        try:
            write_result(_PROTOCOL_OUT, response)
        except Exception:  # noqa: BLE001 - nobody left to report the failure to
            break


if __name__ == "__main__":
    main()
