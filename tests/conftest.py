"""Shared fixtures."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.llm.base import LLMProvider, LLMResponse
from src.sandbox.executor import SandboxExecutor


@pytest.fixture(scope="session")
def sample_df() -> pd.DataFrame:
    """Small deterministic frame with mixed dtypes and real NaNs."""
    rng = np.random.default_rng(0)
    n = 120
    return pd.DataFrame(
        {
            "Age": rng.integers(18, 70, n),
            "Fare": rng.normal(50, 15, n).round(2),
            "Score": np.where(rng.random(n) < 0.15, np.nan, rng.random(n).round(3)),
            "Gender": rng.choice(["Male", "Female"], n),
            "Survived": rng.integers(0, 2, n),
        }
    )


@pytest.fixture(scope="module")
def executor():
    """
    A real sandbox executor with a short timeout.

    Module-scoped: spawning a worker costs a few seconds, so the tests in a
    module share one. Torn down explicitly so no process outlives the run.
    """
    ex = SandboxExecutor(timeout=10)
    yield ex
    ex.shutdown()


class FakeProvider(LLMProvider):
    """Scripted provider so agent tests never touch the network."""

    name = "fake"

    def __init__(self, replies: list[str], model: str = "fake-model"):
        self.model = model
        self.replies = list(replies)
        self.calls: list[list] = []

    def complete(self, system, messages, temperature=0.0, max_tokens=2048) -> LLMResponse:
        index = min(len(self.calls), len(self.replies) - 1)
        self.calls.append(list(messages))
        return LLMResponse(
            text=self.replies[index],
            provider=self.name,
            model=self.model,
            input_tokens=100,
            output_tokens=50,
            latency_s=0.1,
            cost_usd=0.001,
        )

    def is_retryable(self, exc: Exception) -> bool:
        return False


@pytest.fixture
def fake_provider():
    return FakeProvider
