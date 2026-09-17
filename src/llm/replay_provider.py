"""
Offline provider that replays recorded model responses.

Why this exists: without it, the app is a dead end for anyone who opens it
without credentials. `app.py` used to call `settings.validate_provider()` and
`st.stop()`, so the deployed demo would break the day the API key was rotated
or the free-tier quota tripped — which is exactly when a reviewer is most
likely to be looking at it.

What it does NOT do is fake the product. Only the *network call* is replayed.
The response still goes through code extraction, the AST validator, and real
execution in the sandbox worker, so every number and chart a visitor sees was
computed from the actual CSV at the moment they asked. The UI labels the mode
so nobody mistakes a replay for live generation.

The fixture is written by `python -m eval.run_benchmark --record`, and only
from tasks that the benchmark scored *correct*. So the recorded text is
genuine model output that is verified-good by construction.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from src.llm.base import LLMError, LLMProvider, LLMResponse, Message, normalize_messages
from src.logging_conf import get_logger

logger = get_logger(__name__)

DEFAULT_FIXTURE = Path(__file__).resolve().parents[2] / "demo_data" / "recorded_responses.json"

# `agent.generate_and_execute` builds the final turn as
#     f"Dataset:\n{context}\n\nQuestion: {query}"
# so the question can be recovered exactly rather than guessed at.
_QUESTION_MARKER = re.compile(r"\bQuestion:\s*(.+)\s*\Z", re.DOTALL)

_PUNCTUATION = re.compile(r"[^a-z0-9\s]+")
_WHITESPACE = re.compile(r"\s+")

# Jaccard overlap required before a reworded question is accepted.
#
# Chosen from measurement, not taste. Matching only ever runs within one
# dataset, so the value that matters is how similar two *recorded* questions
# for the same dataset get — that is the confusion the threshold has to stay
# above. Across the shipped fixture the worst such pair is 0.53:
#
#   C2 "Which contract type has the highest churn rate, and what is that rate?"
#   C7 "Which internet service type has the highest average monthly charges,
#       and what is that average?"
#
# Meanwhile a user who shortens a suggested question scores around 0.67
# ("what is the total sales revenue" against the recorded "...across all
# orders"). 0.6 sits between the two with room on both sides. Set it too high
# and demo mode refuses questions it plainly knows; too low and it confidently
# answers a different question than the one asked, which is the worse failure.
# `test_replay.py` pins both ends.
_MATCH_THRESHOLD = 0.6


def normalise(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace."""
    return _WHITESPACE.sub(" ", _PUNCTUATION.sub(" ", text.lower())).strip()


def extract_question(messages: list[Message]) -> str | None:
    """Recover the user's question from the last user turn."""
    for message in reversed(messages):
        if message.role != "user":
            continue
        match = _QUESTION_MARKER.search(message.content)
        return match.group(1).strip() if match else message.content.strip()
    return None


@dataclass(frozen=True)
class Recording:
    dataset: str
    question: str
    response: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    task_id: str | None = None

    @property
    def tokens(self) -> frozenset[str]:
        return frozenset(normalise(self.question).split())


def load_recordings(path: Path | None = None) -> list[Recording]:
    """Read the fixture. A missing or unreadable file yields no recordings."""
    fixture = path or DEFAULT_FIXTURE
    if not fixture.exists():
        return []
    try:
        payload = json.loads(fixture.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        logger.warning("Could not read demo fixture at %s", fixture)
        return []

    return [
        Recording(
            dataset=entry["dataset"],
            question=entry["question"],
            response=entry["response"],
            model=entry.get("model", "recorded"),
            input_tokens=entry.get("input_tokens", 0),
            output_tokens=entry.get("output_tokens", 0),
            task_id=entry.get("task_id"),
        )
        for entry in payload.get("entries", [])
    ]


def questions_for(dataset: str, path: Path | None = None) -> list[str]:
    """Questions the UI can offer as one-click examples in demo mode."""
    return [r.question for r in load_recordings(path) if r.dataset == dataset]


class ReplayProvider(LLMProvider):
    """Serves recorded responses for the bundled demo datasets."""

    name = "replay"

    def __init__(
        self,
        dataset: str | None = None,
        fixture_path: Path | None = None,
        recordings: list[Recording] | None = None,
    ):
        self._recordings = (
            recordings if recordings is not None else load_recordings(fixture_path)
        )
        self._dataset = dataset
        # Report the model that actually produced the text, so the UI and any
        # logs attribute it correctly instead of inventing a "replay" model.
        recorded_model = next((r.model for r in self._recordings), "recorded")
        super().__init__(recorded_model)

    @property
    def dataset(self) -> str | None:
        return self._dataset

    def available_questions(self) -> list[str]:
        return [r.question for r in self._in_scope()]

    def _in_scope(self) -> list[Recording]:
        """Recordings valid for the active dataset.

        Scoping matters: the recorded code was written against one specific
        schema. Replaying `customer_churn` code onto `retail_orders` would run
        real code against the wrong columns and fail confusingly.
        """
        if self._dataset is None:
            return list(self._recordings)
        return [r for r in self._recordings if r.dataset == self._dataset]

    def _match(self, question: str) -> Recording | None:
        candidates = self._in_scope()
        if not candidates:
            return None

        target = normalise(question)
        for recording in candidates:
            if normalise(recording.question) == target:
                return recording

        # Fall back to word overlap so trivial rewording still lands.
        asked = frozenset(target.split())
        if not asked:
            return None

        best, best_score = None, 0.0
        for recording in candidates:
            known = recording.tokens
            union = asked | known
            score = len(asked & known) / len(union) if union else 0.0
            if score > best_score:
                best, best_score = recording, score

        return best if best_score >= _MATCH_THRESHOLD else None

    def complete(
        self,
        system: str,
        messages: list[Message | dict],
        temperature: float = 0.0,
        max_tokens: int = 2048,
    ) -> LLMResponse:
        question = extract_question(normalize_messages(messages))
        if question is None:
            raise LLMError("Demo mode received no question to answer.")

        recording = self._match(question)
        if recording is None:
            raise LLMError(self._no_match_message())

        logger.info("Demo mode replayed recording for %r", recording.question)
        return LLMResponse(
            text=recording.response,
            provider=self.name,
            model=recording.model,
            input_tokens=recording.input_tokens,
            output_tokens=recording.output_tokens,
            latency_s=0.0,
            cost_usd=0.0,  # nothing was called, so nothing was spent
        )

    def _no_match_message(self) -> str:
        if not self._recordings:
            return (
                "Demo mode has no recorded answers available. "
                "Set GROQ_API_KEY to generate answers live."
            )
        return (
            "That question isn't part of the offline demo. Demo mode replays a "
            "fixed set of recorded answers for the sample datasets — pick one of "
            "the suggested questions, or set GROQ_API_KEY to ask anything about "
            "your own data."
        )

    @staticmethod
    def is_retryable(exc: Exception) -> bool:
        return False  # replaying is deterministic; retrying changes nothing
