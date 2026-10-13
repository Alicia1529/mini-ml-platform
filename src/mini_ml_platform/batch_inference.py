"""The same batch function, called directly or scheduled as a Ray task."""

import math
import time

import ray

from mini_ml_platform.moderation import Message, ModerationResult, moderate_message


def split_batches(messages: list[Message], batch_size: int) -> list[list[Message]]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    return [messages[start : start + batch_size] for start in range(0, len(messages), batch_size)]


def validate_delay(delay_seconds: float) -> None:
    if not math.isfinite(delay_seconds) or delay_seconds < 0:
        raise ValueError("delay_seconds must be finite and non-negative")


def moderate_batch(messages: list[Message], delay_seconds: float = 0) -> list[ModerationResult]:
    """Sleep once PER MESSAGE so total simulated work is independent of batching."""
    validate_delay(delay_seconds)
    results = []
    for message in messages:
        if delay_seconds:
            time.sleep(delay_seconds)
        results.append(moderate_message(message))
    return results


@ray.remote(num_cpus=1, max_retries=0)
def moderate_batch_task(
    messages: list[Message], delay_seconds: float = 0
) -> list[ModerationResult]:
    """One batch occupies one logical CPU slot until this task completes."""
    return moderate_batch(messages, delay_seconds)


def run_sequential(
    messages: list[Message], batch_size: int, delay_seconds: float = 0
) -> list[ModerationResult]:
    validate_delay(delay_seconds)
    batches = split_batches(messages, batch_size)
    return [result for batch in batches for result in moderate_batch(batch, delay_seconds)]


def run_ray(
    messages: list[Message], batch_size: int, delay_seconds: float = 0
) -> list[ModerationResult]:
    """Caller owns ray.init()/shutdown(); preserve input order when gathering."""
    validate_delay(delay_seconds)
    batches = split_batches(messages, batch_size)
    if not ray.is_initialized():
        raise RuntimeError("Call ray.init(num_cpus=...) before run_ray()")
    # Submit every batch before waiting, allowing Ray to schedule them concurrently.
    refs = [moderate_batch_task.remote(batch, delay_seconds) for batch in batches]
    batch_results = ray.get(refs)
    return [result for batch in batch_results for result in batch]
