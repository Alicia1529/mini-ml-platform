"""Milestone 2 exercise: compare Task and Actor batch inference."""

from time import perf_counter, sleep

import ray

from mini_ml_platform.actor_inference import ModelWorker, run_with_actors
from mini_ml_platform.batch_inference import (
    moderate_batch,
    run_sequential,
    split_batches,
    validate_delay,
)
from mini_ml_platform.moderation import Message, ModerationResult, generate_messages


@ray.remote(num_cpus=1)
def moderate_batch_with_load(
    batch: list[Message], load_seconds: float, delay_seconds: float
) -> list[ModerationResult]:
    """Simulate loading a model for each Task invocation."""
    sleep(load_seconds)
    return moderate_batch(batch, delay_seconds)


def run_with_tasks(
    messages: list[Message],
    batch_size: int,
    load_seconds: float = 0.0,
    delay_seconds: float = 0.0,
) -> list[ModerationResult]:
    validate_delay(load_seconds)
    validate_delay(delay_seconds)
    batches = split_batches(messages, batch_size)

    refs = [
        moderate_batch_with_load.remote(batch, load_seconds, delay_seconds)
        for batch in batches
    ]
    batch_results = ray.get(refs)
    return [result for batch in batch_results for result in batch]


if __name__ == "__main__":
    # Run from the repository root:
    # .venv/bin/python examples/compare_tasks_actors.py
    messages = generate_messages(count=100, seed=42)
    batch_size = 10
    num_cpus = 2
    worker_count = 2
    delay_seconds = 0.05  # Same simulated inference time in every mode.
    load_seconds = 1.0  # One load sequentially, per Task, or per Actor.

    sequential_load_started = perf_counter()
    sleep(load_seconds)
    sequential_load_elapsed = perf_counter() - sequential_load_started

    sequential_work_started = perf_counter()
    expected = run_sequential(messages, batch_size, delay_seconds)
    sequential_work_elapsed = perf_counter() - sequential_work_started

    ray_started = perf_counter()
    ray.init(address="local", num_cpus=num_cpus, include_dashboard=False)
    ray_init_elapsed = perf_counter() - ray_started
    try:
        # Run Tasks before creating Actors, which hold their CPU slots.
        tasks_started = perf_counter()
        task_results = run_with_tasks(messages, batch_size, load_seconds, delay_seconds)
        tasks_elapsed = perf_counter() - tasks_started

        setup_started = perf_counter()
        workers = [ModelWorker.remote(load_seconds) for _ in range(worker_count)]
        ray.get([worker.stats.remote() for worker in workers])  # Wait for model loading.
        actor_setup_elapsed = perf_counter() - setup_started

        actors_started = perf_counter()
        actor_results = run_with_actors(workers, messages, batch_size, delay_seconds)
        actor_first_elapsed = perf_counter() - actors_started

        reuse_started = perf_counter()
        actor_reuse_results = run_with_actors(workers, messages, batch_size, delay_seconds)
        actor_reuse_elapsed = perf_counter() - reuse_started

        assert task_results == actor_results == actor_reuse_results == expected
        batch_count = len(split_batches(messages, batch_size))
        print(f"Messages: {len(messages)}, batches: {batch_count}, Ray CPUs: {num_cpus}")
        print(f"Correct results: {len(expected)}, flagged: {sum(r.flagged for r in expected)}")

        print("\nSequential (one simulated model load)")
        print(f"  Simulated model load: {sequential_load_elapsed:.3f}s")
        print(f"  Batch processing:     {sequential_work_elapsed:.3f}s")
        print(f"  Total:                {sequential_load_elapsed + sequential_work_elapsed:.3f}s")

        print(f"\nRay Tasks ({batch_count} simulated loads, one per batch)")
        print(f"  Ray startup:          {ray_init_elapsed:.3f}s")
        print(f"  Task execution:       {tasks_elapsed:.3f}s (loads + batches + Ray overhead)")
        print(f"  Sum for one pass:     {ray_init_elapsed + tasks_elapsed:.3f}s")

        print(f"\nRay Actors ({worker_count} simulated loads, one per Actor)")
        print(f"  Ray startup:          {ray_init_elapsed:.3f}s")
        print(f"  Actor setup:          {actor_setup_elapsed:.3f}s (includes model loads)")
        print(f"  First batch pass:     {actor_first_elapsed:.3f}s")
        actor_first_total = ray_init_elapsed + actor_setup_elapsed + actor_first_elapsed
        print(f"  Sum for one pass:     {actor_first_total:.3f}s")
        print(f"  Reuse batch pass:     {actor_reuse_elapsed:.3f}s (same Actors, no new load)")
        print("  Batch counts after both passes:", ray.get([worker.stats.remote() for worker in workers]))
        print("\nRay startup was measured once; the Ray modes ran in sequence.")
        print("Each sum combines its measured phases, not a separate cold-start run.")
    finally:
        ray.shutdown()
