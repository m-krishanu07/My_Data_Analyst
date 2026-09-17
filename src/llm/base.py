"""
Provider-agnostic LLM interface.

The agent depends only on this abstraction, so swapping Groq for Bedrock
(or adding another vendor) requires no change to prompting, parsing, or the
self-correction loop.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


class LLMError(RuntimeError):
    """Provider call failed after exhausting retries."""


@dataclass(frozen=True)
class Message:
    """One conversation turn. `role` is 'user' or 'assistant'."""

    role: str
    content: str

    def as_dict(self) -> dict:
        return {"role": self.role, "content": self.content}


@dataclass
class LLMResponse:
    """Normalised completion result, identical in shape across providers."""

    text: str
    provider: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    latency_s: float = 0.0
    cost_usd: float = 0.0
    raw: dict = field(default_factory=dict, repr=False)

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


def normalize_messages(messages: list[Message | dict]) -> list[Message]:
    """
    Coerce to Message objects and enforce a strictly alternating
    user/assistant sequence beginning with 'user'.

    Bedrock rejects consecutive same-role turns outright, and Groq degrades
    on them. Conversation history assembled from a live chat can easily
    violate this (e.g. two assistant messages after a retry), so we merge
    adjacent same-role turns rather than letting the API 400.
    """
    typed = [m if isinstance(m, Message) else Message(m["role"], m["content"]) for m in messages]
    typed = [m for m in typed if m.content and m.content.strip()]

    if not typed:
        return []

    # Drop any leading assistant turns — a conversation must open with user.
    while typed and typed[0].role != "user":
        typed.pop(0)

    merged: list[Message] = []
    for message in typed:
        if merged and merged[-1].role == message.role:
            merged[-1] = Message(
                role=message.role,
                content=f"{merged[-1].content}\n\n{message.content}",
            )
        else:
            merged.append(message)
    return merged


class LLMProvider(ABC):
    """Base class for chat-completion backends."""

    name: str = "base"

    def __init__(self, model: str):
        self.model = model

    @abstractmethod
    def complete(
        self,
        system: str,
        messages: list[Message | dict],
        temperature: float = 0.0,
        max_tokens: int = 2048,
    ) -> LLMResponse:
        """Run a chat completion and return a normalised response."""

    @staticmethod
    def is_retryable(exc: Exception) -> bool:
        """Whether a failed call is worth retrying (rate limits, transient 5xx)."""
        text = str(exc).lower()
        markers = (
            "rate limit", "ratelimit", "429", "too many requests",
            "timeout", "timed out", "connection", "throttl",
            "500", "502", "503", "504", "overloaded", "unavailable",
        )
        return any(marker in text for marker in markers)

    def __repr__(self) -> str:  # pragma: no cover
        return f"{type(self).__name__}(model={self.model!r})"
