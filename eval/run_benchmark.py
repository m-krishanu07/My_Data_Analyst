"""
Execution-based benchmark.

    python -m eval.run_benchmark --verify
    python -m eval.run_benchmark --models groq:llama-3.3-70b-versatile
    python -m eval.run_benchmark --models groq:llama-3.1-8b-instant \
                                          groq:llama-3.3-70b-versatile
    python -m eval.run_benchmark --models groq:llama-3.3-70b-versatile --record

Every task runs through the *real* pipeline — prompt, LLM, AST validator,
sandboxed subprocess, self-correction loop — and is scored on the value that
came back, not on what the generated source looked like.

`--verify` runs each task's reference implementation instead of calling a
model. It costs nothing and must be 18/18; anything less means a check is
wrong, not that a model is bad. Run it before trusting a benchmark number.

`--record` additionally writes `demo_data/recorded_responses.json` from the
answers this run scored *correct*, which is what the app replays when no API
key is configured. Tying the fixture to the benchmark is deliberate: demo mode
can only ever show output that was verified against ground truth.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from eval import datasets
from eval.tasks import TASKS, TASKS_BY_ID, Task
from src.agent.agent import AgentResult, generate_and_execute
from src.config import ConfigError, settings
from src.llm.factory import get_provider
from src.llm.pricing import format_cost
from src.sandbox.executor import get_executor

RESULTS_DIR = Path(__file__).resolve().parent / "results"
RECORDINGS_PATH = Path(__file__).resolve().parents[1] / "demo_data" / "recorded_responses.json"

# Outcomes, ordered worst-to-best for stable reporting.
OUTCOMES = ("timeout", "rejected", "runtime", "llm", "no_code", "no_figure", "wrong", "pass")

OUTCOME_LABELS = {
    "pass": "correct",
    "wrong": "wrong answer",
    "no_figure": "no chart returned",
    "no_code": "no code generated",
    "runtime": "runtime error",
    "rejected": "blocked by validator",
    "timeout": "timed out",
    "llm": "provider error",
}


@dataclass
class TaskRun:
    model: str
    task_id: str
    dataset: str
    category: str
    outcome: str
    attempts: int
    repaired: bool
    input_tokens: int
    output_tokens: int
    latency_s: float
    cost_usd: float
    wall_s: float
    error: str = ""

    @property
    def passed(self) -> bool:
        return self.outcome == "pass"


@dataclass
class ModelReport:
    model: str
    runs: list[TaskRun] = field(default_factory=list)
    # Verbatim model replies for tasks that scored `pass`, for `--record`.
    # Kept off TaskRun so the CSV stays a table of numbers.
    recordings: list[dict] = field(default_factory=list)

    @property
    def total(self) -> int:
        return len(self.runs)

    @property
    def passed(self) -> int:
        return sum(1 for run in self.runs if run.passed)

    @property
    def pass_rate(self) -> float:
        return self.passed / self.total if self.total else 0.0

    @property
    def repaired(self) -> int:
        """Answers that were only correct because the repair loop kicked in."""
        return sum(1 for run in self.runs if run.passed and run.repaired)

    @property
    def total_cost(self) -> float:
        return sum(run.cost_usd for run in self.runs)

    @property
    def median_latency(self) -> float:
        return statistics.median([run.latency_s for run in self.runs]) if self.runs else 0.0

    def breakdown(self) -> dict[str, int]:
        counts = dict.fromkeys(OUTCOMES, 0)
        for run in self.runs:
            counts[run.outcome] += 1
        return {name: count for name, count in counts.items() if count}


# ── Scoring ───────────────────────────────────────────────────


def score(task: Task, result: AgentResult, df) -> str:
    """Classify one answer. Failure reasons come from the agent, not regex."""
    if not result.ok:
        return result.failure_kind or "runtime"
    if result.generated_code is None:
        # The model chatted instead of analysing.
        return "no_code"
    if task.expects_figure and result.figure is None:
        return "no_figure"
    try:
        correct = task.check(result.result_text, df)
    except Exception as exc:  # noqa: BLE001 - a broken check must not abort the run
        print(f"    ! check for {task.id} raised {type(exc).__name__}: {exc}", file=sys.stderr)
        return "wrong"
    return "pass" if correct else "wrong"


# ── Runners ───────────────────────────────────────────────────


def run_reference(task: Task) -> TaskRun:
    """Execute the task's own reference implementation — the harness self-test."""
    df = datasets.load(task.dataset)
    started = time.perf_counter()
    execution = get_executor().execute(task.reference, df)
    wall = time.perf_counter() - started

    result = AgentResult(
        generated_code=task.reference,
        result_text=execution.result_text,
        figure=execution.figure,
        figure_type=execution.figure_type,
        error=execution.error,
        attempts=1,
    )
    if execution.rejected:
        result.failure_kind = "rejected"
    elif execution.timed_out:
        result.failure_kind = "timeout"
    elif execution.error:
        result.failure_kind = "runtime"

    return TaskRun(
        model="reference",
        task_id=task.id,
        dataset=task.dataset,
        category=task.category,
        outcome=score(task, result, df),
        attempts=1,
        repaired=False,
        input_tokens=0,
        output_tokens=0,
        latency_s=0.0,
        cost_usd=0.0,
        wall_s=round(wall, 2),
        error=(execution.error or "")[:300],
    )


def run_task(task: Task, provider) -> tuple[TaskRun, AgentResult]:
    """Run one task. The AgentResult comes back too so `--record` can keep it."""
    df = datasets.load(task.dataset)
    started = time.perf_counter()
    # No conversation history: each task must stand on its own, otherwise a
    # later task could be carried by an earlier answer.
    result = generate_and_execute(task.question, df, provider=provider)
    wall = time.perf_counter() - started

    run = TaskRun(
        model=f"{provider.name}:{provider.model}",
        task_id=task.id,
        dataset=task.dataset,
        category=task.category,
        outcome=score(task, result, df),
        attempts=result.attempts,
        repaired=result.repaired,
        input_tokens=result.input_tokens,
        output_tokens=result.output_tokens,
        latency_s=round(result.latency_s, 2),
        cost_usd=result.cost_usd,
        wall_s=round(wall, 2),
        error=(result.error or "")[:300],
    )
    return run, result


def run_model(spec: str, tasks: list[Task]) -> ModelReport:
    provider_name, _, model_id = spec.partition(":")
    provider = get_provider(provider_name, model_id or None)
    report = ModelReport(model=f"{provider.name}:{provider.model}")

    print(f"\n{report.model}")
    for index, task in enumerate(tasks, 1):
        run, result = run_task(task, provider)
        report.runs.append(run)
        if run.passed and result.raw_response:
            report.recordings.append(
                {
                    "task_id": task.id,
                    "dataset": task.dataset,
                    "question": task.question,
                    "response": result.raw_response,
                    "model": provider.model,
                    "input_tokens": result.input_tokens,
                    "output_tokens": result.output_tokens,
                }
            )
        flag = "PASS" if run.passed else OUTCOME_LABELS[run.outcome].upper()
        note = " (self-corrected)" if run.repaired and run.passed else ""
        print(
            f"  [{index:2d}/{len(tasks)}] {task.id:<4} {flag:<20}"
            f" {run.wall_s:5.1f}s  {format_cost(run.cost_usd)}{note}"
        )
        if not run.passed and run.error:
            print(f"        {run.error.splitlines()[0][:110]}")
    return report


# ── Reporting ─────────────────────────────────────────────────


def write_csv(reports: list[ModelReport], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [asdict(run) for report in reports for run in report.runs]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_recordings(reports: list[ModelReport], path: Path) -> int:
    """
    Write the offline-demo fixture from this run's *correct* answers.

    Only tasks that scored `pass` are recorded, so demo mode can never replay
    model output that the benchmark did not verify against ground truth. That
    provenance is the whole point: the fixture is genuine model text that is
    correct by construction, not prose written by hand and presented as a
    model response.

    Caveat, stated rather than hidden: for a task that only became correct
    after the repair loop, the recorded reply is the *successful* attempt. The
    demo therefore shows a clean first try for a question the model initially
    got wrong. The benchmark table is where the self-correction count lives.

    Returns the number of entries written.
    """
    seen: set[tuple[str, str]] = set()
    entries = []
    for report in reports:
        for entry in report.recordings:
            key = (entry["dataset"], entry["question"])
            if key in seen:
                continue  # first model to get a task right owns that question
            seen.add(key)
            entries.append(entry)

    entries.sort(key=lambda e: (e["dataset"], e["task_id"]))
    payload = {
        "generated": datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC"),
        "note": (
            "Recorded by `python -m eval.run_benchmark --record`. Only tasks the "
            "benchmark scored correct are included. Replayed offline by "
            "src/llm/replay_provider.py; the code is still validated and executed."
        ),
        "entries": entries,
    }

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return len(entries)


def _category_table(reports: list[ModelReport]) -> list[str]:
    categories = sorted({run.category for report in reports for run in report.runs})
    header = "| Category | " + " | ".join(r.model for r in reports) + " |"
    rule = "|---" * (len(reports) + 1) + "|"
    lines = [header, rule]
    for category in categories:
        cells = []
        for report in reports:
            subset = [run for run in report.runs if run.category == category]
            hits = sum(1 for run in subset if run.passed)
            cells.append(f"{hits}/{len(subset)}" if subset else "—")
        lines.append(f"| {category} | " + " | ".join(cells) + " |")
    return lines


def write_markdown(reports: list[ModelReport], path: Path) -> None:
    stamp = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    task_count = reports[0].total if reports else 0

    lines = [
        "# Benchmark",
        "",
        f"_Generated {stamp} by `python -m eval.run_benchmark`._",
        "",
        f"{task_count} analysis tasks over two seeded datasets "
        f"({', '.join(datasets.DATASETS)}). Each task runs the full pipeline — "
        "prompt, LLM, AST validation, sandboxed execution, self-correction — and "
        "is scored on the **value the code returned**, checked against ground "
        "truth recomputed from the dataframe. Generated source is never "
        "string-matched.",
        "",
        "## Results",
        "",
        "| Model | pass@1 | Correct | Self-corrected | Median latency | Cost / query |",
        "|---|---|---|---|---|---|",
    ]

    for report in reports:
        mean_cost = report.total_cost / report.total if report.total else 0.0
        lines.append(
            f"| `{report.model}` | {report.pass_rate:.0%} | "
            f"{report.passed}/{report.total} | {report.repaired} | "
            f"{report.median_latency:.1f}s | {format_cost(mean_cost)} |"
        )

    lines += ["", "## By category", ""]
    lines += _category_table(reports)

    lines += ["", "## Failure taxonomy", ""]
    for report in reports:
        failures = {
            name: count
            for name, count in report.breakdown().items()
            if name != "pass"
        }
        if not failures:
            lines.append(f"- `{report.model}`: no failures.")
            continue
        detail = ", ".join(f"{OUTCOME_LABELS[name]}: {count}" for name, count in failures.items())
        lines.append(f"- `{report.model}`: {detail}")

    lines += [
        "",
        "## Reading this honestly",
        "",
        "- `pass@1` is a single greedy sample per task at `temperature=0`; it is "
        "not an average over repeated runs.",
        "- **Self-corrected** counts answers that were wrong on the first attempt "
        "and only became correct after the repair loop fed the execution error "
        "back to the model. It is the value the loop adds, isolated.",
        "- Numeric answers match if a number within tolerance appears in the "
        "output, so a correct value buried in a table counts. Chart tasks are "
        "scored on whether a renderable figure crossed the sandbox boundary.",
        "- `python -m eval.run_benchmark --verify` runs each task's reference "
        "implementation through the same scoring path and must be "
        f"{task_count}/{task_count}. That is what makes a low score attributable "
        "to the model rather than to a broken check.",
        "- **Latency is wall-clock, and on a free tier it is mostly queueing.** "
        "The timer wraps the whole agent call, so provider-side rate-limit "
        "backoff (the client sleeps and retries on HTTP 429) lands inside the "
        "measurement. These numbers describe what a user on a shared free key "
        "experiences; they are not a measure of model speed and should not be "
        "read as a speed comparison between models.",
        f"- {task_count} tasks over two datasets is a smoke test, not a "
        "benchmark suite. If models tie at the top, the honest reading is that "
        "the suite is too easy to separate them — not that they are equally "
        "capable.",
        "",
    ]

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def print_summary(reports: list[ModelReport]) -> None:
    print("\n" + "=" * 62)
    for report in reports:
        mean_cost = report.total_cost / report.total if report.total else 0.0
        print(
            f"{report.model:<44} {report.passed:>2}/{report.total} "
            f"({report.pass_rate:.0%})"
        )
        print(
            f"{'':<44} median {report.median_latency:.1f}s, "
            f"{format_cost(mean_cost)}/query, "
            f"{format_cost(report.total_cost)} total"
        )
        failures = {k: v for k, v in report.breakdown().items() if k != "pass"}
        if failures:
            print(
                f"{'':<44} "
                + ", ".join(f"{OUTCOME_LABELS[k]}: {v}" for k, v in failures.items())
            )
    print("=" * 62)


# ── CLI ───────────────────────────────────────────────────────


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--models",
        nargs="+",
        metavar="PROVIDER:MODEL",
        help="e.g. groq:llama-3.3-70b-versatile bedrock:us.anthropic.claude-sonnet-4-5-v1:0",
    )
    parser.add_argument(
        "--tasks",
        help="Comma-separated task ids to run (default: all).",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Run reference implementations instead of a model. No API calls.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=RESULTS_DIR,
        help="Where to write benchmark.md / benchmark.csv.",
    )
    parser.add_argument(
        "--no-write",
        action="store_true",
        help="Print results without overwriting the committed report.",
    )
    parser.add_argument(
        "--record",
        nargs="?",
        type=Path,
        const=RECORDINGS_PATH,
        metavar="PATH",
        help="Also write the offline-demo fixture from answers scored correct.",
    )
    return parser.parse_args(argv)


def select_tasks(spec: str | None) -> list[Task]:
    if not spec:
        return TASKS
    selected = []
    for task_id in (part.strip().upper() for part in spec.split(",") if part.strip()):
        if task_id not in TASKS_BY_ID:
            raise SystemExit(f"Unknown task id {task_id!r}. Known: {', '.join(TASKS_BY_ID)}")
        selected.append(TASKS_BY_ID[task_id])
    return selected


def verify(tasks: list[Task]) -> int:
    """Self-test: references must score 100%. Returns a process exit code."""
    print(f"Verifying {len(tasks)} reference implementations against their own checks\n")
    failures = []
    for task in tasks:
        run = run_reference(task)
        status = "ok" if run.passed else OUTCOME_LABELS[run.outcome]
        print(f"  {task.id:<4} {status:<22} {run.wall_s:5.2f}s")
        if not run.passed:
            failures.append(run)
            if run.error:
                print(f"       {run.error}")

    if failures:
        print(f"\n{len(failures)} reference(s) failed: {', '.join(r.task_id for r in failures)}")
        print("The benchmark is not trustworthy until these pass.")
        return 1

    print(f"\nAll {len(tasks)} references pass. Ground truth is sound.")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    tasks = select_tasks(args.tasks)

    try:
        if args.verify:
            return verify(tasks)

        specs = args.models or [f"{settings.provider}:{settings.model_for()}"]
        reports = []
        for spec in specs:
            try:
                reports.append(run_model(spec, tasks))
            except ConfigError as exc:
                # A missing Bedrock key should skip that row, not lose the run.
                print(f"\nSkipping {spec}: {exc}", file=sys.stderr)

        if not reports:
            print("No models could be benchmarked.", file=sys.stderr)
            return 1

        print_summary(reports)

        if not args.no_write:
            write_csv(reports, args.out_dir / "benchmark.csv")
            write_markdown(reports, args.out_dir / "benchmark.md")
            print(f"\nWrote {args.out_dir / 'benchmark.md'} and benchmark.csv")

        if args.record:
            count = write_recordings(reports, args.record)
            print(f"Recorded {count} verified answer(s) to {args.record}")
        return 0
    finally:
        # The warm worker is a child process; leaving it running would hang
        # the interpreter on exit.
        get_executor().shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
