"""Provider construction and retry policy."""

from __future__ import annotations

import time

from src.config import ConfigError, settings
from src.llm.base import LLMError, LLMProvider, LLMResponse, Message
from src.logging_conf import get_logger

logger = get_logger(__name__)

_PROVIDERS = ("groq", "bedrock")

# Providers are cached per (name, model): constructing a boto3 client or
# Groq client on every Streamlit rerun is wasteful.
_cache: dict[tuple[str, str], LLMProvider] = {}


def get_provider(name: str | None = None, model: str | None = None) -> LLMProvider:
    """Build (or reuse) a provider instance."""
    provider_name = (name or settings.provider).lower()
    if provider_name not in _PROVIDERS:
        raise ConfigError(
            f"Unknown provider '{provider_name}'. Supported: {', '.join(_PROVIDERS)}."
        )

    model_id = model or settings.model_for(provider_name)
    key = (provider_name, model_id)

    if key not in _cache:
        if provider_name == "groq":
            from src.llm.groq_provider import GroqProvider

            _cache[key] = GroqProvider(model=model_id)
        else:
            from src.llm.bedrock_provider import BedrockProvider

            _cache[key] = BedrockProvider(model=model_id)
        logger.info("Initialised %s provider (model=%s)", provider_name, model_id)

    return _cache[key]


def complete_with_retry(
    provider: LLMProvider,
    system: str,
    messages: list[Message | dict],
    temperature: float | None = None,
    max_tokens: int | None = None,
    max_retries: int | None = None,
) -> LLMResponse:
    """
    Call the provider, retrying transient failures with exponential backoff.

    Only retries errors classified as transient (rate limits, 5xx, timeouts);
    a bad model id or auth failure fails immediately rather than making the
    user wait through three pointless backoffs.
    """
    agent = settings.agent
    attempts = max_retries or agent.max_api_retries
    temp = agent.temperature if temperature is None else temperature
    tokens = max_tokens or agent.max_tokens

    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            return provider.complete(
                system=system,
                messages=messages,
                temperature=temp,
                max_tokens=tokens,
            )
        except LLMError as exc:
            last_error = exc
            if not provider.is_retryable(exc) or attempt == attempts - 1:
                break
            delay = agent.retry_base_delay * (2**attempt)
            logger.warning(
                "%s call failed (attempt %d/%d), retrying in %.1fs: %s",
                provider.name, attempt + 1, attempts, delay, exc,
            )
            time.sleep(delay)

    raise LLMError(str(last_error) if last_error else "LLM call failed.")
