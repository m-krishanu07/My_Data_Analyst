"""
Parent-side sandbox controller.

Owns a long-lived worker process and enforces the wall-clock timeout that
the previous thread-based implementation could not.

Why a process and not a thread: `thread.join(timeout)` returns when the
timeout expires but the thread keeps running. `while True: pass` therefore
leaked a CPU-bound thread that competed with Streamlit for the GIL for the
lifetime of the app. A process can simply be killed.

The worker is kept warm between queries because importing pandas, sklearn
and plotly costs several seconds, and Windows only supports `spawn` (no
fork), so a cold process per query would dominate response time. On timeout
the worker is destroyed and replaced, which costs one cold start — the
correct trade for a guaranteed stop.
"""

from __future__ import annotations

import contextlib
import os
import queue
import subprocess
import sys
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from src.config import settings
from src.logging_conf import get_logger
from src.sandbox.protocol import read_result, write_job
from src.sandbox.validator import CodeValidationError, validate_code

logger = get_logger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_IS_WINDOWS = os.name == "nt"

# Resolved absolutely rather than left to PATH lookup: we invoke this to kill a
# process that just ran untrusted code, and a `taskkill.exe` planted earlier in
# PATH would turn the cleanup step into the escape.
_TASKKILL = os.path.join(
    os.environ.get("SYSTEMROOT", r"C:\Windows"), "System32", "taskkill.exe"
)


@dataclass
class ExecutionResult:
    """Outcome of running generated code."""

    result_text: str | None = None
    figure: Any = None           # PNG bytes, or a plotly JSON spec string
    figure_type: str | None = None  # "png" | "plotly_json" | None
    stdout: str | None = None
    warning: str | None = None
    error: str | None = None
    timed_out: bool = False
    rejected: bool = False       # blocked by the validator, never executed

    @property
    def ok(self) -> bool:
        return self.error is None

    def as_dict(self) -> dict:
        return {
            "result_text": self.result_text,
            "figure": self.figure,
            "figure_type": self.figure_type,
            "stdout": self.stdout,
            "warning": self.warning,
            "error": self.error,
            "timed_out": self.timed_out,
            "rejected": self.rejected,
        }


@dataclass
class _Worker:
    process: subprocess.Popen
    responses: queue.Queue[Any] = field(default_factory=queue.Queue)
    reader: threading.Thread | None = None
    memory_limited: bool = False


def _kill_process_tree(process: subprocess.Popen) -> None:
    """
    Kill the worker and anything it spawned.

    `process.kill()` alone can orphan grandchildren, which would keep
    consuming CPU after we've reported a timeout.
    """
    if process.poll() is not None:
        return
    try:
        if _IS_WINDOWS:
            subprocess.run(  # noqa: S603 - fixed argv, absolute path, no shell
                [_TASKKILL, "/F", "/T", "/PID", str(process.pid)],
                capture_output=True,
                check=False,
                timeout=10,
            )
        else:
            os.killpg(os.getpgid(process.pid), 9)
    except Exception:  # noqa: BLE001 - best effort, fall through to kill()
        # Worth a log line: a tree-kill that keeps failing means orphaned
        # grandchildren are surviving timeouts, which is a real leak.
        logger.debug("Tree-kill of pid %s failed; falling back to kill().", process.pid)
    finally:
        # The tree-kill may already have reaped it; kill()/wait() are the
        # backstop and either raising must not prevent the other from running.
        with contextlib.suppress(Exception):
            process.kill()
        with contextlib.suppress(Exception):
            process.wait(timeout=5)


class SandboxExecutor:
    """
    Thread-safe façade over a single warm worker process.

    Executions are serialised by a lock. One worker is sufficient because
    each query is interactive and short; it also bounds total memory use.
    """

    def __init__(self, timeout: int | None = None, memory_mb: int | None = None):
        self._timeout = timeout or settings.sandbox.timeout_seconds
        self._memory_mb = memory_mb or settings.sandbox.memory_limit_mb
        self._worker: _Worker | None = None
        self._lock = threading.Lock()

    # ── Worker lifecycle ──────────────────────────────────────

    def _spawn(self) -> _Worker:
        env = os.environ.copy()
        env["SANDBOX_MEMORY_MB"] = str(self._memory_mb)
        env["PYTHONPATH"] = str(_PROJECT_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
        env["PYTHONUNBUFFERED"] = "1"
        env["MPLBACKEND"] = "Agg"

        # A new process group / session is what makes tree-kill possible.
        creation_kwargs: dict = {}
        if _IS_WINDOWS:
            creation_kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
        else:
            creation_kwargs["start_new_session"] = True

        process = subprocess.Popen(
            [sys.executable, "-u", "-m", "src.sandbox.worker"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            cwd=str(_PROJECT_ROOT),
            env=env,
            **creation_kwargs,
        )

        worker = _Worker(process=process)
        worker.reader = threading.Thread(
            target=self._pump, args=(worker,), daemon=True, name="sandbox-reader"
        )
        worker.reader.start()

        handshake = self._await_response(worker, settings.sandbox.startup_timeout_seconds)
        if not isinstance(handshake, dict) or not handshake.get("ready"):
            _kill_process_tree(process)
            raise RuntimeError("Sandbox worker failed to start.")

        worker.memory_limited = bool(handshake.get("memory_limited"))
        logger.info(
            "Sandbox worker started (pid=%s, memory_limit=%s)",
            process.pid,
            f"{self._memory_mb}MB" if worker.memory_limited else "not enforced on this OS",
        )
        return worker

    def _pump(self, worker: _Worker) -> None:
        """Move worker output onto a queue so the main thread can time-box reads."""
        stream = worker.process.stdout
        # A kill closes the pipe mid-frame; that is the normal path out of this
        # loop, not an error worth surfacing. `_EOF` tells the waiter either way.
        with contextlib.suppress(Exception):
            while True:
                message = read_result(stream)
                if message is None:
                    break
                worker.responses.put(message)
        worker.responses.put(_EOF)

    @staticmethod
    def _await_response(worker: _Worker, timeout: float) -> Any:
        try:
            message = worker.responses.get(timeout=timeout)
        except queue.Empty:
            return _TIMEOUT
        return message

    def _ensure_worker(self) -> _Worker:
        if self._worker is None or self._worker.process.poll() is not None:
            if self._worker is not None:
                logger.warning("Sandbox worker died; restarting.")
                _kill_process_tree(self._worker.process)
            self._worker = self._spawn()
        return self._worker

    def _discard_worker(self) -> None:
        if self._worker is not None:
            _kill_process_tree(self._worker.process)
            self._worker = None

    def shutdown(self) -> None:
        with self._lock:
            if self._worker is None:
                return
            # Ask nicely first so the worker can exit cleanly; if the pipe is
            # already gone, _discard_worker kills it regardless.
            with contextlib.suppress(Exception):
                write_job(self._worker.process.stdin, {"shutdown": True})
                self._worker.process.wait(timeout=3)
            self._discard_worker()

    # ── Public API ────────────────────────────────────────────

    def execute(
        self,
        code: str,
        df: pd.DataFrame,
        timeout: int | None = None,
    ) -> ExecutionResult:
        """Validate, then run `code` against `df` in the isolated worker."""
        # Gate 1: static validation. Runs in the parent so malicious code is
        # never handed to an interpreter at all.
        try:
            validate_code(code)
        except CodeValidationError as exc:
            logger.warning("Rejected generated code: %s", exc)
            return ExecutionResult(error=str(exc), rejected=True)

        effective_timeout = timeout or self._timeout

        with self._lock:
            try:
                worker = self._ensure_worker()
            except Exception as exc:
                logger.exception("Could not start sandbox worker")
                return ExecutionResult(error=f"Could not start the sandbox: {exc}")

            job = {
                "code": code,
                "df": df,
                "max_output_chars": settings.sandbox.max_output_chars,
                "max_figure_bytes": settings.sandbox.max_figure_bytes,
            }

            try:
                write_job(worker.process.stdin, job)
            except (BrokenPipeError, OSError):
                self._discard_worker()
                return ExecutionResult(error="Sandbox connection lost. Please retry.")

            response = self._await_response(worker, effective_timeout)

            # Gate 2: the timeout is only meaningful because we can kill.
            if response is _TIMEOUT:
                logger.warning("Execution exceeded %ss; killing worker.", effective_timeout)
                self._discard_worker()
                return ExecutionResult(
                    error=(
                        f"Execution timed out after {effective_timeout} seconds. "
                        "Try narrowing the question or working on a smaller subset."
                    ),
                    timed_out=True,
                )

            if response is _EOF:
                self._discard_worker()
                return ExecutionResult(
                    error=(
                        "The sandbox stopped unexpectedly — this usually means the "
                        "code exhausted available memory."
                    )
                )

        if not isinstance(response, dict):
            return ExecutionResult(error="Malformed response from the sandbox.")

        return ExecutionResult(
            result_text=response.get("result_text"),
            figure=response.get("figure"),
            figure_type=response.get("figure_type"),
            stdout=response.get("stdout"),
            warning=response.get("warning"),
            error=response.get("error"),
        )


# Sentinels distinguishable from any real payload.
_TIMEOUT = object()
_EOF = object()

_default_executor: SandboxExecutor | None = None
_default_lock = threading.Lock()


def get_executor() -> SandboxExecutor:
    """Process-wide singleton, so all Streamlit sessions share one worker."""
    global _default_executor
    with _default_lock:
        if _default_executor is None:
            _default_executor = SandboxExecutor()
        return _default_executor


def safe_execute_code(code: str, df: pd.DataFrame, timeout: int | None = None) -> dict:
    """Backwards-compatible dict-returning wrapper."""
    return get_executor().execute(code, df, timeout=timeout).as_dict()
