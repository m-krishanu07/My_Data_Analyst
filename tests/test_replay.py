"""
Tests for offline demo mode.

Two separate concerns are covered here.

*Matching* is tested against hand-built recordings, so the thresholds are
pinned by examples that read clearly and do not shift when the fixture is
re-recorded.

*Fixture integrity* is tested against the real `demo_data/recorded_responses.json`.
That half is the one that matters: demo mode's whole claim is that only the
network call is replayed and the code still runs for real. If a recorded
response stopped parsing, stopped validating, or stopped executing against its
dataset, the demo would show an error to exactly the audience it exists for.
These tests execute every recorded answer in the real sandbox to prove it does
not.
"""

from __future__ import annotations

import pytest

from eval import datasets
from src.agent.agent import extract_code_and_explanation
from src.llm.base import LLMError, Message
from src.llm.replay_provider import (
    Recording,
    ReplayProvider,
    extract_question,
    load_recordings,
    normalise,
)
from src.sandbox.validator import validate_code

SYSTEM = "you are an analyst"


def user_turn(question: str, context: str = "cols: a, b") -> list[Message]:
    """The exact shape `agent.generate_and_execute` sends."""
    return [Message("user", f"Dataset:\n{context}\n\nQuestion: {question}")]


@pytest.fixture(scope="module")
def fake_recordings() -> list[Recording]:
    return [
        Recording(
            dataset="retail_orders",
            question="What is the total sales revenue across all orders?",
            response="```python\ndef run(df):\n    return (1, None)\n```",
            model="test-model",
            input_tokens=100,
            output_tokens=20,
            task_id="R1",
        ),
        Recording(
            dataset="customer_churn",
            question="What is the overall churn rate?",
            response="```python\ndef run(df):\n    return (2, None)\n```",
            model="test-model",
            task_id="C1",
        ),
    ]


# ── Question extraction ───────────────────────────────────────


def test_question_is_recovered_from_the_agent_turn() -> None:
    messages = user_turn("What is the total revenue?", context="Rows: 500\nCols: a, b")
    assert extract_question(messages) == "What is the total revenue?"


def test_question_falls_back_to_raw_content_without_the_marker() -> None:
    assert extract_question([Message("user", "just asking")]) == "just asking"


def test_only_the_last_user_turn_is_used() -> None:
    messages = [
        Message("user", "Dataset:\nx\n\nQuestion: old question"),
        Message("assistant", "some reply"),
        Message("user", "Dataset:\nx\n\nQuestion: new question"),
    ]
    assert extract_question(messages) == "new question"


def test_normalise_strips_case_and_punctuation() -> None:
    assert normalise("  What IS the Total?! ") == "what is the total"


# ── Matching ──────────────────────────────────────────────────


def test_exact_question_replays(fake_recordings) -> None:
    provider = ReplayProvider(dataset="retail_orders", recordings=fake_recordings)
    response = provider.complete(
        SYSTEM, user_turn("What is the total sales revenue across all orders?")
    )
    assert "def run(df)" in response.text
    assert response.input_tokens == 100


def test_punctuation_and_case_do_not_break_matching(fake_recordings) -> None:
    provider = ReplayProvider(dataset="retail_orders", recordings=fake_recordings)
    response = provider.complete(
        SYSTEM, user_turn("what is the TOTAL sales revenue across all orders")
    )
    assert "return (1, None)" in response.text


def test_close_rewording_still_matches(fake_recordings) -> None:
    provider = ReplayProvider(dataset="retail_orders", recordings=fake_recordings)
    response = provider.complete(SYSTEM, user_turn("What is the total sales revenue?"))
    assert "return (1, None)" in response.text


def test_similar_recorded_questions_do_not_collide() -> None:
    """
    The lower bound on the match threshold. C2 and C7 are the most similar
    pair of recorded questions that share a dataset (~0.53 overlap — both are
    "which X has the highest Y, and what is that Y"). Each must still retrieve
    its own answer rather than the other's.
    """
    provider = ReplayProvider(dataset="customer_churn", recordings=FIXTURE)
    by_id = {r.task_id: r for r in FIXTURE}
    for task_id in ("C2", "C7"):
        expected = by_id[task_id]
        assert provider.complete(SYSTEM, user_turn(expected.question)).text == (
            expected.response
        )


def test_unrelated_question_is_refused_rather_than_guessed(fake_recordings) -> None:
    """Answering the wrong question is worse than declining to answer."""
    provider = ReplayProvider(dataset="retail_orders", recordings=fake_recordings)
    with pytest.raises(LLMError, match="offline demo"):
        provider.complete(SYSTEM, user_turn("Plot a histogram of customer ages"))


def test_recordings_are_scoped_to_the_active_dataset(fake_recordings) -> None:
    """
    Recorded code was written against one schema. Replaying the churn answer
    onto retail data would execute real code against columns that do not
    exist, so scope is enforced before matching.
    """
    provider = ReplayProvider(dataset="retail_orders", recordings=fake_recordings)
    with pytest.raises(LLMError):
        provider.complete(SYSTEM, user_turn("What is the overall churn rate?"))

    churn = ReplayProvider(dataset="customer_churn", recordings=fake_recordings)
    assert "return (2, None)" in churn.complete(
        SYSTEM, user_turn("What is the overall churn rate?")
    ).text


def test_unknown_dataset_offers_nothing(fake_recordings) -> None:
    provider = ReplayProvider(dataset="titanic", recordings=fake_recordings)
    assert provider.available_questions() == []
    with pytest.raises(LLMError, match="offline demo"):
        provider.complete(SYSTEM, user_turn("What is the overall churn rate?"))


def test_empty_fixture_tells_the_user_how_to_fix_it() -> None:
    provider = ReplayProvider(dataset="retail_orders", recordings=[])
    with pytest.raises(LLMError, match="GROQ_API_KEY"):
        provider.complete(SYSTEM, user_turn("anything"))


def test_replay_is_free_and_reports_the_recorded_model(fake_recordings) -> None:
    provider = ReplayProvider(dataset="customer_churn", recordings=fake_recordings)
    response = provider.complete(SYSTEM, user_turn("What is the overall churn rate?"))
    assert response.cost_usd == 0.0
    assert response.provider == "replay"
    # The text came from a real model; attribute it to that model, not "replay".
    assert response.model == "test-model"


def test_retrying_a_replay_is_pointless(fake_recordings) -> None:
    assert ReplayProvider.is_retryable(LLMError("boom")) is False


def test_missing_fixture_file_yields_no_recordings(tmp_path) -> None:
    assert load_recordings(tmp_path / "absent.json") == []


def test_unreadable_fixture_does_not_crash_the_app(tmp_path) -> None:
    broken = tmp_path / "broken.json"
    broken.write_text("{not json", encoding="utf-8")
    assert load_recordings(broken) == []


# ── The shipped fixture ───────────────────────────────────────

FIXTURE = load_recordings()


def test_fixture_is_present_and_covers_both_sample_datasets() -> None:
    assert FIXTURE, (
        "demo_data/recorded_responses.json is empty or missing. Regenerate it:\n"
        "    python -m eval.run_benchmark --record"
    )
    assert {r.dataset for r in FIXTURE} == set(datasets.DATASETS)


def test_fixture_questions_are_unique_per_dataset() -> None:
    """A duplicate would make the first match win silently."""
    keys = [(r.dataset, normalise(r.question)) for r in FIXTURE]
    assert len(keys) == len(set(keys))


@pytest.mark.parametrize(
    "recording", FIXTURE, ids=[f"{r.dataset}-{r.task_id}" for r in FIXTURE]
)
def test_recorded_response_parses_and_validates(recording: Recording) -> None:
    code, _ = extract_code_and_explanation(recording.response)
    assert code, f"{recording.task_id}: no code block in the recorded response"
    validate_code(code)  # raises CodeValidationError on a policy violation


@pytest.mark.parametrize(
    "recording", FIXTURE, ids=[f"{r.dataset}-{r.task_id}" for r in FIXTURE]
)
def test_recorded_response_executes_against_its_dataset(
    executor, recording: Recording
) -> None:
    """
    The load-bearing test for demo mode's honesty claim: a visitor without an
    API key sees numbers computed from the CSV right now, not recorded numbers.
    That is only true if every recorded answer still runs.
    """
    code, _ = extract_code_and_explanation(recording.response)
    result = executor.execute(code, datasets.load(recording.dataset), timeout=30)

    assert not result.rejected, f"{recording.task_id}: rejected by validator"
    assert not result.timed_out, f"{recording.task_id}: timed out"
    assert result.error is None, f"{recording.task_id}: {result.error}"
    assert result.result_text or result.figure, f"{recording.task_id}: produced nothing"


def test_offscript_question_surfaces_the_explanation_not_an_outage() -> None:
    """
    An unmatched question is an expected state in demo mode, not a failure of
    the provider. The user must see the explanation verbatim rather than
    "Could not reach replay: …", which reads as if the service were down.
    """
    from src.agent.agent import generate_and_execute

    provider = ReplayProvider(dataset="retail_orders", recordings=FIXTURE)
    result = generate_and_execute(
        "What is the airspeed velocity of an unladen swallow?",
        datasets.load("retail_orders"),
        provider=provider,
    )
    assert not result.ok
    assert "could not reach" not in (result.error or "").lower()
    assert "offline demo" in (result.error or "")


def test_every_fixture_question_matches_itself() -> None:
    """
    Round-trip: the questions the UI shows as clickable chips must be the ones
    the matcher accepts. A threshold change that broke this would make demo
    mode refuse its own suggestions.
    """
    for recording in FIXTURE:
        provider = ReplayProvider(dataset=recording.dataset, recordings=FIXTURE)
        response = provider.complete(SYSTEM, user_turn(recording.question))
        assert response.text == recording.response
