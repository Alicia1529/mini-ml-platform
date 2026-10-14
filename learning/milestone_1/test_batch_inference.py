import pytest
import ray

from batch_inference import (
    generate_messages,
    moderate_batch_task,
    run_parallel,
    run_sequential,
    split_batches,
)


def test_messages_are_reproducible():
    messages = generate_messages()
    assert len(messages) == 1_000
    assert messages == generate_messages(seed=42)
    assert messages != generate_messages(seed=43)


def test_batches_keep_every_message_in_order():
    messages = ["one", "two", "three", "four", "five"]
    batches = split_batches(messages, 2)
    assert batches == [["one", "two"], ["three", "four"], ["five"]]
    assert split_batches([], 2) == []


def test_invalid_inputs_are_rejected():
    with pytest.raises(ValueError, match="count"):
        generate_messages(-1)
    with pytest.raises(ValueError, match="batch_size"):
        split_batches([], 0)
    with pytest.raises(ValueError, match="delay_seconds"):
        run_sequential([], 1, -1)
    with pytest.raises(ValueError, match="delay_seconds"):
        run_parallel([], 1, float("nan"))


def test_sequential_has_independently_known_results():
    messages = ["hello", "SCAM", "nice", "send PASSWORD", "hackathon"]
    assert run_sequential(messages, 2) == [False, True, False, True, True]


@pytest.fixture(scope="module")
def ray_runtime():
    ray.init(address="local", num_cpus=2, include_dashboard=False)
    yield
    ray.shutdown()


@pytest.mark.integration
@pytest.mark.parametrize(("count", "batch_size"), [(0, 3), (11, 3), (1_000, 50)])
def test_ray_matches_sequential(ray_runtime, count, batch_size):
    messages = generate_messages(count)
    assert run_parallel(messages, batch_size) == run_sequential(messages, batch_size)


@pytest.mark.integration
def test_worker_error_reaches_driver(ray_runtime):
    ref = moderate_batch_task.remote(["hello"], -1)
    with pytest.raises(ray.exceptions.RayTaskError, match="delay_seconds"):
        ray.get(ref)
