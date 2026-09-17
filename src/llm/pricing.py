"""
Token pricing, so every response carries a real cost figure.

Rates are USD per 1M tokens and are published list prices at the time of
writing. They drift — treat reported costs as close estimates, not billing.
Unknown models price at zero rather than guessing.
"""

from __future__ import annotations

# model id -> (input $/1M tokens, output $/1M tokens)
PRICING: dict[str, tuple[float, float]] = {
    # ── Groq ──
    "llama-3.3-70b-versatile": (0.59, 0.79),
    "llama-3.1-8b-instant": (0.05, 0.08),
    "llama-3.1-70b-versatile": (0.59, 0.79),
    "openai/gpt-oss-120b": (0.15, 0.75),
    "openai/gpt-oss-20b": (0.10, 0.50),
    "moonshotai/kimi-k2-instruct": (1.00, 3.00),
    "qwen/qwen3-32b": (0.29, 0.59),
    # ── AWS Bedrock (Anthropic) ──
    "us.anthropic.claude-sonnet-4-5-20250929-v1:0": (3.00, 15.00),
    "us.anthropic.claude-haiku-4-5-20251001-v1:0": (1.00, 5.00),
    "us.anthropic.claude-3-5-sonnet-20241022-v2:0": (3.00, 15.00),
    "us.anthropic.claude-3-5-haiku-20241022-v1:0": (0.80, 4.00),
    # ── AWS Bedrock (Meta / Amazon) ──
    "us.meta.llama3-3-70b-instruct-v1:0": (0.72, 0.72),
    "us.amazon.nova-pro-v1:0": (0.80, 3.20),
    "us.amazon.nova-lite-v1:0": (0.06, 0.24),
}

_PER_MILLION = 1_000_000


def estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """Return estimated USD cost for one call. Unknown models cost 0.0."""
    rates = PRICING.get(model)
    if rates is None:
        # Bedrock ids are region-prefixed (us./eu./apac.); try the bare id.
        for prefix in ("us.", "eu.", "apac."):
            if model.startswith(prefix):
                rates = PRICING.get(model[len(prefix):])
                break
    if rates is None:
        return 0.0

    input_rate, output_rate = rates
    return (input_tokens * input_rate + output_tokens * output_rate) / _PER_MILLION


def format_cost(cost_usd: float) -> str:
    """Human-friendly cost string; sub-cent values need more precision."""
    if cost_usd <= 0:
        return "—"
    if cost_usd < 0.01:
        return f"${cost_usd:.5f}"
    return f"${cost_usd:.4f}"
