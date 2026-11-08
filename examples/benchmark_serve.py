"""Benchmark throughput and request latency for a running Ray Serve endpoint."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from time import perf_counter
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import numpy as np


def send_one(url: str) -> tuple[bool, float]:
    """POST one SCAM message; return correctness and request duration in seconds."""
    started = perf_counter()
    payload = '{"text":"SCAM"}'
    request = Request(
        url,
        data=payload.encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=10) as response:
            body = json.load(response)
            succeeded = (
                response.status == 200
                and isinstance(body, dict)
                and body.get("flagged") is True
            )
    except (HTTPError, URLError, TimeoutError, json.JSONDecodeError):
        succeeded = False
    return succeeded, perf_counter() - started


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark the Serve moderation API")
    parser.add_argument("--requests", type=int, default=20)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--url", default="http://127.0.0.1:8000/")
    args = parser.parse_args()

    if args.requests <= 0 or args.concurrency <= 0:
        parser.error("--requests and --concurrency must be positive")

    started = perf_counter()
    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        results = list(executor.map(send_one, [args.url] * args.requests))
    elapsed = perf_counter() - started

    succeeded = sum(ok for ok, _ in results)
    print(f"Requests: {args.requests}, succeeded: {succeeded}, failed: {args.requests - succeeded}")
    print(f"Total time: {elapsed:.3f}s")
    print(f"Throughput: {succeeded / elapsed:.2f} successful requests/s")
    successful_latencies_ms = [seconds * 1000 for ok, seconds in results if ok]
    if successful_latencies_ms:
        p50, p99 = np.percentile(successful_latencies_ms, [50, 99])
        print(f"Successful request latency: p50={p50:.1f}ms, p99={p99:.1f}ms")
    else:
        print("Successful request latency: n/a (no successful requests)")


if __name__ == "__main__":
    main()
