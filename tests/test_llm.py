"""
LLM layer: message normalisation, cost accounting, retry policy.

No network calls — providers are stubbed. The normalisation tests matter
because Bedrock's `converse` API rejects payloads that Groq tolerates.
"""

from __future__ import annotations

import pytest

from src.config import ConfigError
from src.llm.base import LLMError, LLMProvider, LLMResponse, Message, normalize_messages
from src.llm.factory import complete_with_retry, get_provider
from src.llm.pricing import estimate_cost, format_cost

# ── Message normalisation ─────────────────────────────────────


def test_accepts_dicts_and_messages() -> None:
    result = normalize_messages([{"role": "user", "content": "hi"}, Message("assistant", "yo")])
    assert all(isinstance(m, Message) for m in result)
    assert [m.role for m in result] == ["user", "assistant"]


def test_merges_consecutive_same_role_turns() -> None:
    """Bedrock rejects two user turns in a row; Groq does not. Normalise for both."""
    result = normalize_messages(
        [Message("user", "first"), Message("user", "second"), Message("assistant", "reply")]
    )
    assert [m.role for m in result] == ["user", "assistant"]
    assert "first" in result[0].content and "second" in result[0].content


def test_drops_leading_assistant_turn() -> None:
    """The conversation must start with `user`."""
    result = normalize_messages([Message("assistant", "stray"), Message("user", "real")])
    assert result[0].role == "user"
    assert result[0].content == "real"


def test_empty_history_is_allowed() -> None:
    assert normalize_messages([]) == []


def test_blank_content_is_dropped() -> None:
    result = normalize_messages([Message("user", "   "), Message("user", "real")])
    assert len(result) == 1
    assert result[0].content == "real"


# ── Cost accounting ───────────────────────────────────────────


def test_known_model_cost_is_computed() -> None:
    cost = estimate_cost("llama-3.3-70b-versatile", 1_000_000, 1_000_000)
    assert cost > 0


def test_cost_scales_with_tokens() -> None:
    small = estimate_cost("llama-3.3-70b-versatile", 1000, 500)
    large = estimate_cost("llama-3.3-70b-versatile", 10_000, 5000)
    assert large == pytest.approx(10 * small)


def test_bedrock_region_prefix_falls_back_to_base_model() -> None:
    """`us.anthropic...` is an inference profile ID, not a separate price."""
    assert estimate_cost("us.anthropic.claude-sonnet-4-5-20250929-v1:0", 1000, 1000) > 0


def test_unknown_model_costs_zero_rather_than_raising() -> None:
    assert estimate_cost("some-model-we-have-never-seen", 1000, 1000) == 0.0


def test_cost_formatting_is_readable() -> None:
    assert "$" in format_cost(0.0123)
    # Sub-cent costs need extra precision or every query reads as "$0.00".
    assert format_cost(0.00098) == "$0.00098"


def test_zero_cost_renders_as_a_dash_not_a_fake_zero() -> None:
    """Unknown pricing must not be displayed as a confident '$0.00'."""
    assert "$" not in format_cost(0.0)


def test_response_totals_tokens() -> None:
    response = LLMResponse(text="x", provider="p", model="m", input_tokens=10, output_tokens=5)
    assert response.total_tokens == 15


# ── Retry policy ──────────────────────────────────────────────


class _FlakyProvider(LLMProvider):
    name = "flaky"

    def __init__(self, failures: int, retryable: bool = True):
        self.model = "m"
        self.failures = failures
        self.retryable = retryable
        self.attempts = 0

    def complete(self, system, messages, temperature=0.0, max_tokens=2048):
        self.attempts += 1
        if self.attempts <= self.failures:
            raise LLMError("rate limit exceeded (429)")
        return LLMResponse(text="ok", provider=self.name, model=self.model)

    def is_retryable(self, exc):
        return self.retryable


def test_transient_failures_are_retried(monkeypatch) -> None:
    monkeypatch.setattr("time.sleep", lambda _: None)  # don't actually back off
    provider = _FlakyProvider(failures=2)
    assert complete_with_retry(provider, "sys", [Message("user", "hi")]).text == "ok"
    assert provider.attempts == 3


def test_permanent_failures_are_not_retried(monkeypatch) -> None:
    """A bad API key must fail fast instead of three pointless backoffs."""
    monkeypatch.setattr("time.sleep", lambda _: None)
    provider = _FlakyProvider(failures=99, retryable=False)
    with pytest.raises(LLMError):
        complete_with_retry(provider, "sys", [Message("user", "hi")])
    assert provider.attempts == 1


def test_retries_are_bounded(monkeypatch) -> None:
    monkeypatch.setattr("time.sleep", lambda _: None)
    provider = _FlakyProvider(failures=99)
    with pytest.raises(LLMError):
        complete_with_retry(provider, "sys", [Message("user", "hi")], max_retries=3)
    assert provider.attempts == 3


# ── Factory ───────────────────────────────────────────────────


def test_unknown_provider_is_rejected() -> None:
    with pytest.raises(ConfigError, match="Unknown provider"):
        get_provider("not-a-real-provider")


def test_missing_groq_key_gives_actionable_message(monkeypatch) -> None:
    """C3: this used to be a cryptic crash at import time."""
    from src.config import Settings

    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    with pytest.raises(ConfigError) as exc:
        Settings(groq_api_key=None).validate_provider("groq")
    message = str(exc.value)
    assert "GROQ_API_KEY" in message
    assert "console.groq.com" in message


def test_available_providers_reports_usable_ones(monkeypatch) -> None:
    from src.config import Settings

    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    assert "groq" in Settings(groq_api_key="test-key").available_providers()


def test_bedrock_without_credentials_is_rejected(monkeypatch) -> None:
    """Having boto3 installed is not the same as being able to call Bedrock."""
    from src.config import Settings

    monkeypatch.setattr("src.config._aws_credentials_available", lambda region: False)
    with pytest.raises(ConfigError, match="AWS credentials"):
        Settings().validate_provider("bedrock")


def test_bedrock_with_credentials_is_accepted(monkeypatch) -> None:
    from src.config import Settings

    monkeypatch.setattr("src.config._aws_credentials_available", lambda region: True)
    assert "bedrock" in Settings(groq_api_key=None).available_providers()
