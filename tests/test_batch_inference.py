import pytest
import ray

from mini_ml_platform.batch_inference import (
    moderate_batch,
    moderate_batch_task,
    run_ray,
    run_sequential,
    split_batches,
)
from mini_ml_platform.moderation import Message, ModerationResult, generate_messages, moderate_message


def test_generation_is_reproducible_with_unique_ids():
    messages = generate_messages()
    assert len(messages) == 1_000
    assert messages == generate_messages(seed=42)
    assert messages != generate_messages(seed=43)
    assert [message.id for message in messages] == list(range(1_000))
    assert any(moderate_message(message).flagged for message in messages)
    assert any(not moderate_message(message).flagged for message in messages)
    assert generate_messages(0) == []
    with pytest.raises(ValueError, match="count"):
        generate_messages(-1)


@pytest.mark.parametrize(("text", "matches"), [
    ("Nice game!", ()),
    ("", ()),
    ("SCAM, password, HaCk! scam", ("scam", "password", "hack")),
    ("Hackathon", ("hack",)),  # Deliberate substring semantics, including false positives.
])
def test_moderation_rules(text, matches):
    assert moderate_message(Message(17, text)) == ModerationResult(17, bool(matches), matches)


@pytest.mark.parametrize(("count", "size", "lengths"), [
    (0, 3, []), (5, 2, [2, 2, 1]), (4, 2, [2, 2]), (3, 10, [3]), (3, 1, [1, 1, 1]),
])
def test_batches_preserve_every_message_in_order(count, size, lengths):
    messages = generate_messages(count)
    batches = split_batches(messages, size)
    assert [len(batch) for batch in batches] == lengths
    assert [message for batch in batches for message in batch] == messages


@pytest.mark.parametrize("size", [0, -1])
def test_invalid_batch_size(size):
    with pytest.raises(ValueError, match="batch_size"):
        split_batches([], size)


@pytest.mark.parametrize("delay", [-1, float("nan"), float("inf")])
@pytest.mark.parametrize("runner", [run_sequential, run_ray])
def test_invalid_delay_is_rejected_even_with_empty_input(runner, delay):
    with pytest.raises(ValueError, match="delay_seconds"):
        runner([], batch_size=1, delay_seconds=delay)


def test_simulated_delay_is_per_message(monkeypatch):
    sleeps = []
    monkeypatch.setattr("mini_ml_platform.batch_inference.time.sleep", sleeps.append)
    messages = generate_messages(3)
    assert moderate_batch(messages, 0.02) == [moderate_message(m) for m in messages]
    assert sleeps == [0.02, 0.02, 0.02]


def test_sequential_matches_independent_expected_results():
    messages = [Message(9, "Hello"), Message(3, "SCAM"), Message(8, "password HACK")]
    assert run_sequential(messages, 2) == [
        ModerationResult(9, False, ()),
        ModerationResult(3, True, ("scam",)),
        ModerationResult(8, True, ("password", "hack")),
    ]


def test_ray_requires_explicit_initialization():
    assert not ray.is_initialized()
    with pytest.raises(RuntimeError, match="ray.init"):
        run_ray([], 1)


@pytest.fixture(scope="module")
def ray_runtime():
    ray.init(address="local", num_cpus=2, include_dashboard=False)
    yield
    ray.shutdown()


@pytest.mark.integration
@pytest.mark.parametrize(("count", "size", "delay"), [
    (0, 3, 0), (11, 3, 0), (9, 1, 0.001), (5, 20, 0), (1_000, 50, 0),
])
def test_real_ray_matches_sequential(ray_runtime, count, size, delay):
    messages = generate_messages(count)
    assert run_ray(messages, size, delay) == run_sequential(messages, size, delay)


@pytest.mark.integration
def test_worker_exception_reaches_driver(ray_runtime):
    ref = moderate_batch_task.remote([Message(0, "Hello")], -1)
    with pytest.raises(ray.exceptions.RayTaskError, match="delay_seconds"):
        ray.get(ref)
