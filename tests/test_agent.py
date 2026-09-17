"""
Agent orchestration: code extraction, the self-correction loop, and the
guarantee that raw errors never reach the model's conversation history.
"""

from __future__ import annotations

import pytest

from src.agent.agent import (
    AgentResult,
    build_data_context,
    extract_code_and_explanation,
    generate_and_execute,
    summarize_for_history,
)

pytestmark = pytest.mark.timeout(120)

GOOD = "```python\ndef run(df):\n    return (float(df['Age'].mean()), None)\n```\nThe mean age."
BAD = "```python\ndef run(df):\n    return (df['Missing'].mean(), None)\n```"
ESCAPE = "```python\ndef run(df):\n    return (().__class__.__bases__, None)\n```"


# ── Code extraction ───────────────────────────────────────────


def test_extracts_code_and_prose() -> None:
    code, prose = extract_code_and_explanation(GOOD)
    assert "def run(df):" in code
    assert prose == "The mean age."


def test_prefers_the_block_that_defines_run() -> None:
    """Models often emit a usage example alongside the real answer."""
    text = (
        "Example:\n```python\nrun(df)\n```\n"
        "Answer:\n```python\ndef run(df):\n    return (1, None)\n```"
    )
    code, _ = extract_code_and_explanation(text)
    assert "def run(df):" in code


def test_no_code_block_returns_prose_only() -> None:
    code, prose = extract_code_and_explanation("Hello! Ask me about your data.")
    assert code is None
    assert prose == "Hello! Ask me about your data."


def test_empty_text_is_safe() -> None:
    assert extract_code_and_explanation("") == (None, "")


def test_handles_unlabelled_fence() -> None:
    code, _ = extract_code_and_explanation("```\ndef run(df):\n    return (1, None)\n```")
    assert "def run" in code


# ── Happy path ────────────────────────────────────────────────


def test_successful_query(sample_df, fake_provider) -> None:
    result = generate_and_execute("mean age", sample_df, provider=fake_provider([GOOD]))
    assert result.ok, result.error
    assert result.attempts == 1
    assert not result.repaired
    assert float(result.result_text) == pytest.approx(sample_df["Age"].mean())


def test_greeting_skips_execution(sample_df, fake_provider) -> None:
    provider = fake_provider(["Hi! Upload a CSV and ask me something."])
    result = generate_and_execute("hello", sample_df, provider=provider)
    assert result.ok
    assert result.generated_code is None
    assert "Hi!" in result.explanation


# ── Self-correction ───────────────────────────────────────────


def test_repairs_after_a_failure(sample_df, fake_provider) -> None:
    provider = fake_provider([BAD, GOOD])
    result = generate_and_execute("mean age", sample_df, provider=provider)

    assert result.ok, result.error
    assert result.repaired
    assert result.attempts == 2
    assert len(result.failed_attempts) == 1
    assert "Missing" in result.failed_attempts[0]["error"]


def test_repair_prompt_receives_the_error(sample_df, fake_provider) -> None:
    provider = fake_provider([BAD, GOOD])
    generate_and_execute("mean age", sample_df, provider=provider)

    retry_messages = provider.calls[1]
    assert any("failed" in m.content.lower() for m in retry_messages)
    assert any("Missing" in m.content for m in retry_messages)


def test_gives_up_after_the_configured_attempts(sample_df, fake_provider) -> None:
    """Token spend must be bounded even if the model never recovers."""
    from src.config import settings

    provider = fake_provider([BAD])
    result = generate_and_execute("mean age", sample_df, provider=provider)

    assert not result.ok
    assert result.attempts == 1 + settings.agent.max_repair_attempts


def test_usage_accumulates_across_attempts(sample_df, fake_provider) -> None:
    """Reported cost must be the true cost of answering, not just the last call."""
    single = generate_and_execute("q", sample_df, provider=fake_provider([GOOD]))
    repaired = generate_and_execute("q", sample_df, provider=fake_provider([BAD, GOOD]))

    assert repaired.total_tokens == 2 * single.total_tokens
    assert repaired.cost_usd == pytest.approx(2 * single.cost_usd)


def test_rejected_code_is_reported_not_executed(sample_df, fake_provider) -> None:
    result = generate_and_execute("escape", sample_df, provider=fake_provider([ESCAPE]))
    assert not result.ok
    assert "__class__" in result.error


# ── Q1: history must never contain tracebacks ─────────────────


def test_history_summary_is_short_and_clean() -> None:
    result = AgentResult(
        explanation="The mean is 42.",
        result_text="42.0",
        error=None,
    )
    summary = summarize_for_history(result)
    assert "Traceback" not in summary
    assert len(summary) <= 400
    assert "42" in summary


def test_history_summary_truncates_large_tables() -> None:
    result = AgentResult(explanation="x" * 5000, result_text="y" * 5000)
    assert len(summarize_for_history(result)) <= 400


def test_history_is_passed_to_the_provider(sample_df, fake_provider) -> None:
    provider = fake_provider([GOOD])
    history = [
        {"role": "user", "content": "how many rows?"},
        {"role": "assistant", "content": "120 rows."},
    ]
    generate_and_execute("and columns?", sample_df, conversation_history=history, provider=provider)
    contents = [m.content for m in provider.calls[0]]
    assert "how many rows?" in contents
    assert "120 rows." in contents


# ── Prompt context ────────────────────────────────────────────


def test_data_context_lists_columns_and_dtypes(sample_df) -> None:
    context = build_data_context(sample_df)
    for column in sample_df.columns:
        assert column in context
    assert "rows" in context
    assert "nulls" in context


def test_data_context_is_capped_for_wide_frames() -> None:
    import numpy as np
    import pandas as pd

    wide = pd.DataFrame(np.zeros((3, 300)), columns=[f"col_{i}" for i in range(300)])
    context = build_data_context(wide)
    assert "more columns" in context
    assert len(context) < 20_000


def test_provider_construction_failure_is_reported(sample_df, monkeypatch) -> None:
    """A bad API key must surface as a message, not a crash."""
    import src.agent.agent as agent_module

    def boom():
        raise RuntimeError("GROQ_API_KEY is not set.")

    monkeypatch.setattr(agent_module, "get_provider", boom)
    result = generate_and_execute("q", sample_df)
    assert not result.ok
    assert "GROQ_API_KEY" in result.error
