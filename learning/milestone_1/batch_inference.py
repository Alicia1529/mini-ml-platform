import argparse
import math
from random import Random
from time import perf_counter, sleep

import ray

from moderation import is_unsafe

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
    """Use a local seeded RNG so runs produce the same messages."""
    if count < 0:
        raise ValueError("count must be non-negative")
    rng = Random(seed)
    return [rng.choice(MESSAGE_TEMPLATES) for i in range(count)]


def split_batches(messages: list[str], batch_size: int) -> list[list[str]]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    return [messages[start:start + batch_size] for start in range(0, len(messages), batch_size)]


def validate_delay(delay_seconds: float) -> None:
    if not math.isfinite(delay_seconds) or delay_seconds < 0:
        raise ValueError("delay_seconds must be finite and non-negative")


def moderate_message(message: str, delay_seconds: float) -> bool:
    if delay_seconds:
        sleep(delay_seconds)
    return is_unsafe(message)


def run_sequential(
    messages: list[str], batch_size: int, delay_seconds: float = 0.0
) -> list[bool]:
    validate_delay(delay_seconds)
    results = []

    for batch in split_batches(messages, batch_size):
        for message in batch:
            results.append(moderate_message(message, delay_seconds))

    return results


@ray.remote(num_cpus=1, max_retries=0)
def moderate_batch_task(batch: list[str], delay_seconds: float = 0.0) -> list[bool]:
    validate_delay(delay_seconds)
    results = []
    for message in batch:
        results.append(moderate_message(message, delay_seconds))
    return results


def run_parallel(
    messages: list[str], batch_size: int, delay_seconds: float = 0.0
) -> list[bool]:
    validate_delay(delay_seconds)
    batches = split_batches(messages, batch_size)
    refs = [moderate_batch_task.remote(batch, delay_seconds) for batch in batches]
    batch_results = ray.get(refs)
    return [result for batch in batch_results for result in batch]


if __name__ == "__main__":
    # ../../.venv/bin/python batch_inference.py
    parser = argparse.ArgumentParser(description="Compare sequential and Ray batch inference")
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--cpus", type=int, default=2)
    parser.add_argument("--delay", type=float, default=0.05)
    parser.add_argument("--dashboard", action="store_true", help="keep Ray running until Enter")
    args = parser.parse_args()

    if args.count < 0 or args.batch_size <= 0 or args.cpus <= 0:
        parser.error("count must be non-negative; batch-size and cpus must be positive")
    if not math.isfinite(args.delay) or args.delay < 0:
        parser.error("delay must be finite and non-negative")

    messages = generate_messages(count=args.count)
    print(f"count={args.count}, batch_size={args.batch_size}, cpus={args.cpus}, delay={args.delay}s")
    print("Sleep simulates latency; this is a concurrency demo, not an ML benchmark.")

    start = perf_counter()
    sequential = run_sequential(messages, args.batch_size, args.delay)
    sequential_seconds = perf_counter() - start

    try:
        start = perf_counter()
        context = ray.init(
            address="local", num_cpus=args.cpus,
            include_dashboard=args.dashboard, dashboard_host="127.0.0.1",
        )
        init_seconds = perf_counter() - start

        start = perf_counter()
        parallel = run_parallel(messages, args.batch_size, args.delay)
        parallel_seconds = perf_counter() - start

        if parallel != sequential:
            raise RuntimeError("Sequential and Ray results differ")
        print("Confirm results: True")
        print("Total:", len(parallel), "Unsafe:", sum(parallel))
        print(f"Sequential: {sequential_seconds:.3f} s, {len(messages) / sequential_seconds:.1f} messages/s")
        print(f"Ray init:   {init_seconds:.3f} s")
        print(f"Parallel:   {parallel_seconds:.3f} s, {len(messages) / parallel_seconds:.1f} messages/s")
        print(f"Ray total:  {init_seconds + parallel_seconds:.3f} s (init + execution)")
        if args.dashboard:
            print(f"Dashboard: http://{context.dashboard_url}")
            try:
                input("Inspect the dashboard, then press Enter to shut down Ray. ")
            except EOFError:
                pass
    finally:
        ray.shutdown()
