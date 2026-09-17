"""
End-to-end sandbox tests — these actually spawn worker processes.

The timeout tests are the important ones. The previous thread-based sandbox
reported a timeout but left the runaway thread executing, so `while True:
pass` permanently consumed a core.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from src.sandbox.executor import SandboxExecutor

pytestmark = pytest.mark.timeout(120)


def run(executor, body: str, df, **kw):
    return executor.execute(f"def run(df):\n{body}\n", df, **kw)


# ── Happy path ────────────────────────────────────────────────


def test_scalar_result(executor, sample_df) -> None:
    result = run(executor, "    return (float(df['Age'].mean()), None)", sample_df)
    assert result.ok, result.error
    assert float(result.result_text) == pytest.approx(sample_df["Age"].mean())


def test_dataframe_result_is_rendered_as_table(executor, sample_df) -> None:
    result = run(
        executor,
        "    return (df.groupby('Gender')['Fare'].mean().reset_index(), None)",
        sample_df,
    )
    assert result.ok, result.error
    assert "Gender" in result.result_text


def test_worker_is_reused_between_calls(executor, sample_df) -> None:
    """A cold start costs seconds; the second call must not pay it."""
    run(executor, "    return (1, None)", sample_df)  # warm up
    start = time.perf_counter()
    result = run(executor, "    return (2, None)", sample_df)
    assert result.ok
    assert time.perf_counter() - start < 2.0


# ── S4: the timeout must actually stop execution ──────────────


def test_infinite_loop_is_killed() -> None:
    executor = SandboxExecutor(timeout=3)
    try:
        import pandas as pd

        start = time.perf_counter()
        result = executor.execute(
            "def run(df):\n    while True:\n        pass\n", pd.DataFrame({"a": [1]})
        )
        elapsed = time.perf_counter() - start

        assert result.timed_out, "runaway code was not detected as a timeout"
        assert not result.ok
        assert elapsed < 30, f"timeout did not fire promptly (took {elapsed:.1f}s)"
    finally:
        executor.shutdown()


def test_worker_recovers_after_a_timeout() -> None:
    """A killed worker must be replaced, not leave the app permanently broken."""
    executor = SandboxExecutor(timeout=3)
    try:
        import pandas as pd

        df = pd.DataFrame({"a": [1, 2, 3]})
        assert executor.execute("def run(df):\n    while True:\n        pass\n", df).timed_out
        recovered = executor.execute("def run(df):\n    return ('alive', None)\n", df)
        assert recovered.ok, recovered.error
        assert "alive" in recovered.result_text
    finally:
        executor.shutdown()


def test_killed_worker_leaves_no_process_behind() -> None:
    executor = SandboxExecutor(timeout=3)
    try:
        import pandas as pd

        executor.execute("def run(df):\n    while True:\n        pass\n", pd.DataFrame({"a": [1]}))
        worker = executor._worker
        assert worker is None or worker.process.poll() is not None
    finally:
        executor.shutdown()


# ── C2: non-tuple returns must not raise ──────────────────────


@pytest.mark.parametrize(
    "body,expect_text",
    [
        ("    return 42", "42"),
        ("    return df['Age'].mean()", None),
        ("    return (df.head(), None)", "Age"),
        ("    return None", None),
        ("    return ('only one element',)", "only one element"),
        ("    return {'a': 1}", "a"),
    ],
)
def test_non_tuple_returns_are_handled(executor, sample_df, body, expect_text) -> None:
    result = run(executor, body, sample_df)
    assert result.ok, result.error
    if expect_text:
        assert expect_text in (result.result_text or "")


# ── Figures ───────────────────────────────────────────────────


def test_matplotlib_figure_returns_png(executor, sample_df) -> None:
    result = run(
        executor,
        "    fig, ax = plt.subplots()\n"
        "    ax.hist(df['Age'])\n"
        "    return ('chart', fig)",
        sample_df,
    )
    assert result.ok, result.error
    assert result.figure_type == "png"
    assert isinstance(result.figure, bytes)
    assert result.figure[:4] == b"\x89PNG"


def test_plotly_figure_returns_json_and_rebuilds(executor, sample_df) -> None:
    """
    Plotly crosses the process boundary as JSON, not a rasterised PNG.

    This is what removed the kaleido dependency that used to crash the app,
    and it keeps charts interactive in the browser.
    """
    result = run(executor, "    return ('scatter', px.scatter(df, x='Age', y='Fare'))", sample_df)
    assert result.ok, result.error
    assert result.figure_type == "plotly_json"

    import plotly.io as pio

    figure = pio.from_json(result.figure)
    assert figure.data  # rebuilds into a real Figure


def test_no_figure_returns_none(executor, sample_df) -> None:
    result = run(executor, "    return (1, None)", sample_df)
    assert result.figure is None
    assert result.figure_type is None


# ── Errors and isolation ──────────────────────────────────────


def test_runtime_error_is_short_and_actionable(executor, sample_df) -> None:
    result = run(executor, "    return (df['NoSuchColumn'].mean(), None)", sample_df)
    assert not result.ok
    assert "NoSuchColumn" in result.error
    # A full traceback would leak internals and pollute the model's context.
    assert "Traceback" not in result.error
    assert len(result.error) < 500


def test_validator_rejection_is_flagged_and_never_executes(executor, sample_df) -> None:
    result = run(executor, "    return (().__class__.__bases__, None)", sample_df)
    assert result.rejected
    assert not result.ok


def test_stdout_is_captured_without_corrupting_the_protocol(executor, sample_df) -> None:
    """
    User `print()` must not desynchronise the length-prefixed pickle stream
    that the worker uses to reply.
    """
    result = run(
        executor,
        "    print('hello from user code')\n    return ('done', None)",
        sample_df,
    )
    assert result.ok, result.error
    assert "hello from user code" in (result.stdout or "")
    assert "done" in result.result_text


def test_sandbox_cannot_read_a_real_file(executor, sample_df, tmp_path: Path) -> None:
    secret = tmp_path / "secret.txt"
    secret.write_text("TOP_SECRET_VALUE")
    body = f"    return (open(r'{secret}').read(), None)"
    result = run(executor, body, sample_df)
    assert not result.ok
    assert "TOP_SECRET_VALUE" not in (result.result_text or "")


def test_sandbox_cannot_write_a_file(executor, sample_df, tmp_path: Path) -> None:
    target = tmp_path / "exfiltrated.csv"
    result = run(executor, f"    df.to_csv(r'{target}')\n    return ('written', None)", sample_df)
    assert not result.ok
    assert not target.exists()


def test_mutating_df_does_not_affect_the_caller(executor, sample_df) -> None:
    """The worker gets a pickled copy, so the session DataFrame is safe."""
    before = len(sample_df)
    result = run(executor, "    df.drop(df.index, inplace=True)\n    return (len(df), None)", sample_df)
    assert result.ok, result.error
    assert len(sample_df) == before


# ── Protocol: the parent must never unpickle worker output ────


def test_parent_refuses_to_execute_a_malicious_reply() -> None:
    """
    A compromised worker must not be able to escalate into the web app.

    The parent reads the worker's replies, and the worker is the process
    running LLM-generated code. If that direction were pickled, a worker that
    had broken out could reply with a `__reduce__` payload and get code
    execution in the parent — turning a contained breach into a total one.
    Results travel as JSON, which has no such opcodes, so a hostile reply is
    at worst a parse error.
    """
    import io
    import pickle

    from src.sandbox.protocol import _write_frame, read_result

    executed = []

    class Exploit:
        def __reduce__(self):
            return (executed.append, ("pwned",))

    frame = io.BytesIO()
    _write_frame(frame, pickle.dumps(Exploit()))
    frame.seek(0)

    with pytest.raises(Exception):  # noqa: B017 - any refusal is a pass; silence is not
        read_result(frame)
    assert executed == []


def test_figure_bytes_survive_the_json_hop(executor, sample_df) -> None:
    """PNG bytes are base64-encoded across the pipe and must arrive unchanged."""
    result = run(
        executor,
        "    fig, ax = plt.subplots()\n    ax.plot([1, 2, 3])\n    return (None, fig)",
        sample_df,
    )
    assert result.ok, result.error
    assert isinstance(result.figure, bytes)
    assert result.figure.startswith(b"\x89PNG\r\n\x1a\n")
