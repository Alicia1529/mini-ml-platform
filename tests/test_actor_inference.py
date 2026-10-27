"""Milestone 2 exercise: verify that Actor workers keep state across batches."""

import pytest
import ray

from mini_ml_platform.actor_inference import ModelWorker, run_with_actors
from mini_ml_platform.batch_inference import run_sequential
from mini_ml_platform.moderation import Message


@pytest.fixture(scope="module")
def ray_runtime():
    ray.init(address="local", num_cpus=2, include_dashboard=False)
    yield
    ray.shutdown()


@pytest.mark.integration
def test_actor_reuse_matches_sequential(ray_runtime):
    # .venv/bin/python -m pytest -q -m integration tests/test_actor_inference.py
    texts = ["hello", "SCAM", "nice", "send PASSWORD", "hackathon"]
    messages = [Message(i, text) for i, text in enumerate(texts)]
    workers = [ModelWorker.remote(0), ModelWorker.remote(0)]

    parallel1 = run_with_actors(workers, messages, batch_size=2, delay_seconds=0.0)
    parallel2 = run_with_actors(workers, messages, batch_size=2, delay_seconds=0.0)

    expected = run_sequential(messages, batch_size=2)

    assert parallel1 == expected
    assert parallel2 == expected
    stats = ray.get([worker.stats.remote() for worker in workers])
    assert stats[0]["batches_processed"] == 4
    assert stats[1]["batches_processed"] == 2
