"""Groq chat-completions provider."""

from __future__ import annotations

import time

from src.config import ConfigError, settings
from src.llm.base import LLMError, LLMProvider, LLMResponse, Message, normalize_messages
from src.llm.pricing import estimate_cost
from src.logging_conf import get_logger

logger = get_logger(__name__)


class GroqProvider(LLMProvider):
    name = "groq"

    def __init__(self, model: str | None = None, api_key: str | None = None):
        super().__init__(model or settings.groq_model)
        key = api_key or settings.groq_api_key
        if not key:
            raise ConfigError(
                "GROQ_API_KEY is not set. Add it to .env locally or to "
                "Secrets on Streamlit Cloud. Free keys: https://console.groq.com/keys"
            )
        try:
            from groq import Groq
        except ImportError as exc:  # pragma: no cover
            raise ConfigError("The 'groq' package is not installed. Run: pip install groq") from exc

        self._client = Groq(api_key=key)

    def complete(
        self,
        system: str,
        messages: list[Message | dict],
        temperature: float = 0.0,
        max_tokens: int = 2048,
    ) -> LLMResponse:
        payload = [{"role": "system", "content": system}]
        payload.extend(m.as_dict() for m in normalize_messages(messages))

        started = time.perf_counter()
        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=payload,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        except Exception as exc:
            raise LLMError(str(exc)) from exc
        latency = time.perf_counter() - started

        usage = getattr(response, "usage", None)
        input_tokens = getattr(usage, "prompt_tokens", 0) or 0
        output_tokens = getattr(usage, "completion_tokens", 0) or 0

        return LLMResponse(
            text=response.choices[0].message.content or "",
            provider=self.name,
            model=self.model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_s=latency,
            cost_usd=estimate_cost(self.model, input_tokens, output_tokens),
        )
