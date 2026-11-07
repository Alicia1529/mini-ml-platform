"""Checkpoint 1: benchmark a running Ray Serve HTTP endpoint."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from time import perf_counter
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


def send_one(url: str) -> bool:
    """POST one SCAM message; return True only for the expected response."""
    payload = '{"text":"SCAM"}'
    request = Request(
        url,
        data=payload.encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=10) as response:
            if response.status != 200:
                return False
            body = json.load(response)
    except (HTTPError, URLError, TimeoutError, json.JSONDecodeError):
        return False
    return isinstance(body, dict) and body.get("flagged") is True


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

    succeeded = sum(results)
    print(f"Requests: {args.requests}, succeeded: {succeeded}, failed: {args.requests - succeeded}")
    print(f"Total time: {elapsed:.3f}s")
    print(f"Throughput: {succeeded / elapsed:.2f} successful requests/s")


if __name__ == "__main__":
    main()
