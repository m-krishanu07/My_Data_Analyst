"""
Agent orchestration.

Pipeline:  question -> LLM -> code -> validate -> execute -> answer
                          ^                          |
                          +------ repair on failure --+

The repair loop is what makes this usable in practice. LLM-generated
analysis code fails often for mundane reasons (a mistyped column, a dtype
mismatch, misaligned NaN handling). Feeding the error back lets the model
correct itself instead of surfacing a dead end to the user.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import pandas as pd

from src.agent.prompts import FEW_SHOT_EXAMPLES, REPAIR_PROMPT, SYSTEM_PROMPT
from src.config import settings
from src.llm.base import LLMError, LLMProvider, Message
from src.llm.factory import complete_with_retry, get_provider
from src.logging_conf import get_logger
from src.sandbox.executor import ExecutionResult, get_executor

logger = get_logger(__name__)

_CODE_BLOCK = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.DOTALL)
_MAX_SAMPLE_COLS = 40


@dataclass
class AgentResult:
    """Everything the UI needs to render one turn."""

    explanation: str | None = None
    generated_code: str | None = None
    # The model's reply verbatim, before code extraction. The benchmark's
    # `--record` mode writes this to the demo fixture, so what gets replayed
    # offline is the exact text the model produced — not a reconstruction.
    raw_response: str | None = None
    result_text: str | None = None
    figure: object = None
    figure_type: str | None = None
    stdout: str | None = None
    warning: str | None = None
    error: str | None = None

    attempts: int = 0
    repaired: bool = False
    # Why the answer failed, when it did: "llm" | "rejected" | "timeout" |
    # "runtime". The benchmark's error taxonomy reads this instead of
    # pattern-matching the user-facing error text, which is free to change.
    failure_kind: str | None = None
    provider: str | None = None
    model: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    latency_s: float = 0.0
    cost_usd: float = 0.0

    # Code from earlier failed attempts, kept for the "show your working" view.
    failed_attempts: list[dict] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.error is None

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


# ── Prompt construction ───────────────────────────────────────


def build_data_context(df: pd.DataFrame) -> str:
    """
    Compact schema description for the model.

    Exact column names, dtypes, null counts and a few sample rows prevent
    the most common generation failures (guessed column names, wrong dtype
    assumptions). Column listing is capped so a 500-column upload cannot
    blow up the prompt.
    """
    lines = [f"Shape: {len(df):,} rows x {len(df.columns)} columns", "", "Columns:"]

    for col in df.columns[:_MAX_SAMPLE_COLS]:
        series = df[col]
        detail = f"  - {col!r} ({series.dtype}), {series.isna().sum()} nulls"
        try:
            if pd.api.types.is_numeric_dtype(series) and series.notna().any():
                detail += f", range {series.min():.4g} to {series.max():.4g}"
            elif series.nunique(dropna=True) <= 8:
                sample = series.dropna().unique()[:8]
                detail += f", values: {[str(v) for v in sample]}"
            else:
                detail += f", {series.nunique(dropna=True)} unique"
        except (TypeError, ValueError):
            pass
        lines.append(detail)

    if len(df.columns) > _MAX_SAMPLE_COLS:
        lines.append(f"  … and {len(df.columns) - _MAX_SAMPLE_COLS} more columns")

    lines.append("")
    lines.append("First rows:")
    try:
        lines.append(df.head(3).to_string(index=False, max_cols=15))
    except Exception:  # noqa: BLE001
        lines.append("(preview unavailable)")

    return "\n".join(lines)


def _few_shot_messages() -> list[Message]:
    messages: list[Message] = []
    for example in FEW_SHOT_EXAMPLES:
        messages.append(Message("user", example["question"]))
        messages.append(Message("assistant", example["answer"]))
    return messages


def extract_code_and_explanation(text: str) -> tuple[str | None, str]:
    """
    Split a model reply into code and prose.

    Handles replies with no code (greetings) and replies with multiple
    blocks — in the latter case the block defining `run` wins, since models
    sometimes emit a usage example alongside the real answer.
    """
    if not text:
        return None, ""

    blocks = list(_CODE_BLOCK.finditer(text))
    if not blocks:
        return None, text.strip()

    chosen = next((m for m in blocks if "def run" in m.group(1)), blocks[0])
    code = chosen.group(1).strip()

    prose = (text[: chosen.start()] + "\n" + text[chosen.end():]).strip()
    prose = _CODE_BLOCK.sub("", prose).strip()
    return code, prose


def _summarize_for_history(result: AgentResult, max_chars: int = 400) -> str:
    """
    Condense an answer for the conversation history.

    Only a short natural-language summary is stored. Full tracebacks and
    large tables are deliberately excluded: they previously polluted the
    context window and degraded later turns.
    """
    parts = [p for p in (result.explanation, result.result_text) if p]
    summary = " ".join(parts).strip() or "(completed)"
    summary = re.sub(r"\s+", " ", summary)
    return summary[:max_chars]


# ── Main entry point ──────────────────────────────────────────


def generate_and_execute(
    query: str,
    df: pd.DataFrame,
    conversation_history: list | None = None,
    provider: LLMProvider | None = None,
) -> AgentResult:
    """Answer one question, repairing generated code if it fails."""
    result = AgentResult()

    try:
        llm = provider or get_provider()
    except Exception as exc:  # noqa: BLE001 - config/credential problems
        result.error = str(exc)
        result.failure_kind = "llm"
        return result

    result.provider = llm.name
    result.model = llm.model

    base_messages: list[Message] = _few_shot_messages()
    for turn in conversation_history or []:
        role = turn["role"] if isinstance(turn, dict) else turn.role
        content = turn["content"] if isinstance(turn, dict) else turn.content
        base_messages.append(Message(role, content))

    base_messages.append(
        Message("user", f"Dataset:\n{build_data_context(df)}\n\nQuestion: {query}")
    )

    executor = get_executor()
    max_attempts = 1 + max(0, settings.agent.max_repair_attempts)
    messages = list(base_messages)
    execution: ExecutionResult | None = None

    for attempt in range(1, max_attempts + 1):
        result.attempts = attempt

        try:
            response = complete_with_retry(llm, SYSTEM_PROMPT, messages)
        except LLMError as exc:
            result.error = _friendly_llm_error(exc, llm.name)
            result.failure_kind = "llm"
            return result

        # Accumulate usage across repair attempts so the reported cost is
        # the true cost of answering, not just the last call.
        result.input_tokens += response.input_tokens
        result.output_tokens += response.output_tokens
        result.latency_s += response.latency_s
        result.cost_usd += response.cost_usd

        result.raw_response = response.text
        code, prose = extract_code_and_explanation(response.text)

        # No code block: conversational reply (greeting, clarification).
        if not code:
            result.explanation = prose or "I'm not sure how to answer that."
            return result

        execution = executor.execute(code, df)
        result.generated_code = code
        result.explanation = prose or None

        if execution.ok:
            result.result_text = execution.result_text
            result.figure = execution.figure
            result.figure_type = execution.figure_type
            result.stdout = execution.stdout
            result.warning = execution.warning
            result.repaired = attempt > 1
            if result.repaired:
                logger.info("Self-corrected after %d attempt(s)", attempt)
            return result

        logger.info(
            "Attempt %d/%d failed (%s): %s",
            attempt, max_attempts,
            "rejected" if execution.rejected else "runtime",
            (execution.error or "")[:200],
        )
        result.failed_attempts.append({"code": code, "error": execution.error})

        # A timeout will almost certainly recur, and each retry costs another
        # full timeout window. Stop and tell the user instead.
        if execution.timed_out or attempt == max_attempts:
            break

        messages = [
            *messages,
            Message("assistant", response.text),
            Message("user", REPAIR_PROMPT.format(error=execution.error)),
        ]

    if execution is not None:
        result.error = execution.error
        result.warning = execution.warning
        if execution.timed_out:
            result.failure_kind = "timeout"
        elif execution.rejected:
            result.failure_kind = "rejected"
        else:
            result.failure_kind = "runtime"
    else:  # pragma: no cover - defensive
        result.error = "The assistant could not produce an answer."
        result.failure_kind = "runtime"
    return result


def _friendly_llm_error(exc: Exception, provider_name: str) -> str:
    text = str(exc)
    if provider_name == "replay":
        # Nothing was reached or unreachable — demo mode simply has no recording
        # for this question, and it already says so in the user's own terms.
        # Wrapping it in "Could not reach replay:" would turn a clear
        # explanation into an apparent outage.
        return text
    lowered = text.lower()
    if "rate" in lowered or "429" in lowered:
        return (
            f"{provider_name} is rate-limiting requests right now. "
            "Wait a few seconds and try again."
        )
    if "api key" in lowered or "401" in lowered or "unauthor" in lowered:
        return f"{provider_name} rejected the credentials. Check your API key configuration."
    return f"Could not reach {provider_name}: {text}"


def summarize_for_history(result: AgentResult, max_chars: int = 400) -> str:
    """Public wrapper used by the UI when appending to conversation history."""
    return _summarize_for_history(result, max_chars)
