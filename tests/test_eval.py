"""
The benchmark harness.

Q3: the old `evaluate_models.py` substring-matched generated source against a
hand-written snippet, on an empty DataFrame. Its accuracy number measured
nothing. The replacement scores the value the code actually returned, which
is only trustworthy if the ground-truth checks are themselves correct — so
the reference implementations are executed here, in CI, not just on demand.
"""

from __future__ import annotations

import pytest

from eval import datasets
from eval.run_benchmark import ModelReport, TaskRun, run_reference, score, write_csv, write_markdown
from eval.tasks import TASKS, close, close_rate, exact, mentions, numbers_in
from src.agent.agent import AgentResult

pytestmark = pytest.mark.timeout(180)


@pytest.fixture(scope="module", autouse=True)
def _shutdown_sandbox():
    """`run_reference` uses the process-wide worker; don't leak it."""
    yield
    from src.sandbox.executor import get_executor

    get_executor().shutdown()


# ── Ground truth ──────────────────────────────────────────────


@pytest.mark.parametrize("task", TASKS, ids=[t.id for t in TASKS])
def test_reference_implementation_passes_its_own_check(task) -> None:
    """
    A failure here means the benchmark is broken, not that a model is bad.

    This also exercises the reference code against the AST validator, so a
    task cannot ask for something the sandbox would refuse to run.
    """
    run = run_reference(task)
    assert run.outcome == "pass", f"{task.id}: {run.outcome} — {run.error}"


def test_every_task_targets_a_real_dataset() -> None:
    for task in TASKS:
        assert task.dataset in datasets.DATASETS


def test_task_ids_are_unique() -> None:
    ids = [task.id for task in TASKS]
    assert len(ids) == len(set(ids))


# ── Answer matching ───────────────────────────────────────────


def test_numbers_are_extracted_from_prose_and_tables() -> None:
    assert numbers_in("Total sales: 3,412,880.55") == [3412880.55]
    # Markdown rules must not read as numbers — sandbox output is markdown.
    assert numbers_in("| a | b |\n|---|---|\n| 1 | 2.5 |") == [1.0, 2.5]
    assert numbers_in(None) == []


def test_close_accepts_a_correct_value_anywhere_in_the_output() -> None:
    assert close("The mean is 682.19 for North", 682.19)
    assert not close("The mean is 700.00 for North", 682.19)


def test_close_rate_accepts_fraction_or_percentage() -> None:
    assert close_rate("0.138", 0.138)
    assert close_rate("13.8% were returned", 0.138)
    assert not close_rate("31.0%", 0.138)


def test_exact_rejects_off_by_one_counts() -> None:
    assert exact("412", 412)
    assert not exact("413", 412)


def test_mentions_is_case_insensitive_and_requires_all_tokens() -> None:
    assert mentions("Top: Electronics and Furniture", "electronics", "FURNITURE")
    assert not mentions("Top: Electronics", "Electronics", "Clothing")


# ── Outcome classification ────────────────────────────────────


def _task(task_id: str):
    return next(t for t in TASKS if t.id == task_id)


def test_wrong_answer_is_not_scored_as_an_error() -> None:
    """A confidently wrong number is a different failure from a crash."""
    task = _task("R1")
    result = AgentResult(generated_code="def run(df): ...", result_text="0")
    assert score(task, result, datasets.load(task.dataset)) == "wrong"


def test_failure_kind_drives_the_taxonomy() -> None:
    task = _task("R1")
    for kind in ("timeout", "rejected", "runtime", "llm"):
        result = AgentResult(error="boom", failure_kind=kind)
        assert score(task, result, datasets.load(task.dataset)) == kind


def test_prose_only_reply_is_scored_as_no_code() -> None:
    task = _task("R1")
    result = AgentResult(explanation="Hello!", generated_code=None)
    assert score(task, result, datasets.load(task.dataset)) == "no_code"


def test_chart_task_without_a_figure_fails() -> None:
    task = _task("R8")
    assert task.expects_figure
    result = AgentResult(generated_code="def run(df): ...", result_text="done", figure=None)
    assert score(task, result, datasets.load(task.dataset)) == "no_figure"


def test_a_raising_check_does_not_abort_the_benchmark() -> None:
    """One malformed answer must not take down a 40-minute run."""
    task = _task("R2")
    result = AgentResult(generated_code="def run(df): ...", result_text=object())  # type: ignore[arg-type]
    assert score(task, result, datasets.load(task.dataset)) in {"wrong", "pass"}


# ── Reporting ─────────────────────────────────────────────────


def _run(outcome: str, **kw) -> TaskRun:
    defaults = {
        "model": "groq:test", "task_id": "R1", "dataset": "retail_orders",
        "category": "aggregation", "attempts": 1, "repaired": False,
        "input_tokens": 100, "output_tokens": 50,
        "latency_s": 1.0, "cost_usd": 0.001, "wall_s": 1.2,
    }
    return TaskRun(outcome=outcome, **{**defaults, **kw})


def test_pass_rate_and_repair_credit() -> None:
    report = ModelReport(
        model="groq:test",
        runs=[_run("pass"), _run("pass", repaired=True), _run("wrong"), _run("timeout")],
    )
    assert report.pass_rate == 0.5
    # Repair credit only counts attempts that ended up correct.
    assert report.repaired == 1
    assert report.breakdown() == {"timeout": 1, "wrong": 1, "pass": 2}


def test_report_files_are_written(tmp_path) -> None:
    report = ModelReport(model="groq:test", runs=[_run("pass"), _run("runtime", error="KeyError")])
    write_csv([report], tmp_path / "benchmark.csv")
    write_markdown([report], tmp_path / "benchmark.md")

    csv_text = (tmp_path / "benchmark.csv").read_text(encoding="utf-8")
    assert "outcome" in csv_text and "KeyError" in csv_text

    md_text = (tmp_path / "benchmark.md").read_text(encoding="utf-8")
    assert "pass@1" in md_text
    assert "50%" in md_text
    assert "runtime error: 1" in md_text
