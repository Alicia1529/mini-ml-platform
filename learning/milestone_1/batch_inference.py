from moderation import is_unsafe
from dataclasses import dataclass
from random import Random
from time import perf_counter
import ray


UNSAFE_KEYWORDS = ("scam", "password", "hack")
MESSAGE_TEMPLATES = (
    "Great game, everyone!",
    "Meet me near the blue tower.",
    "Can someone help with this level?",
    "I like your new avatar.",
    "That obstacle course was fun!",
    "This is a SCAM offer.",
    "Send me your password for a reward.",
    "Download this hack to win.",
)


def generate_messages(count: int = 1000, seed: int = 42) -> list[str]:
    """Use a local seeded RNG so runs share the same input and stable IDs."""
    if count < 0:
        raise ValueError("count must be non-negative")
    rng = Random(seed)
    return [rng.choice(MESSAGE_TEMPLATES) for i in range(count)]


def split_batches(messages: list[str], batch_size: int) -> list[list[str]]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    return [ messages[start:start + batch_size] for start in range(0, len(messages), batch_size)]


def run_sequential(messages: list[str], batch_size: int) -> list[bool]:
    results = []

    for batch in split_batches(messages, batch_size):
        for message in batch:
            results.append(is_unsafe(message))

    return results


@ray.remote(num_cpus=1)
def moderate_batch_task(batch: list[str]) -> list[bool]:
    results = []
    for message in batch:
        results.append(is_unsafe(message))
    return results


def run_parallel(messages: list[str], batch_size: int) -> list[bool]:
    batches = split_batches(messages, batch_size)
    refs = [moderate_batch_task.remote(batch) for batch in batches]
    batch_results = ray.get(refs)
    return [result for batch in batch_results for result in batch]


if __name__ == "__main__":
    # ../../.venv/bin/python batch_inference.py
    messages = generate_messages()
    ray.init(address="local", num_cpus=2, include_dashboard=False)

    try:
        start = perf_counter()
        sequential = run_sequential(messages, 50)
        sequential_seconds = perf_counter() - start

        start = perf_counter()
        parallel = run_parallel(messages, 50)
        parallel_seconds = perf_counter() - start

        print("Confirm results:", parallel == sequential)
        print("Total:", len(parallel), "Unsafe:", sum(parallel))
        print(f"Sequential: {sequential_seconds:.6f} s")
        print(f"Parallel:   {parallel_seconds:.6f} s")
    finally:
        ray.shutdown()
