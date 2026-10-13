"""Run from the repository root after installing the project (see README)."""

import argparse
import math
from time import perf_counter

import ray

from mini_ml_platform.batch_inference import run_ray, run_sequential
from mini_ml_platform.moderation import generate_messages


def positive_int(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def nonnegative_float(value: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise argparse.ArgumentTypeError("must be finite and non-negative")
    return number


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare sequential Python with local Ray Tasks.")
    parser.add_argument("--mode", choices=("both", "sequential", "ray"), default="both")
    parser.add_argument("--messages", type=positive_int, default=1_000)
    parser.add_argument("--batch-size", type=positive_int, default=50)
    parser.add_argument("--num-cpus", type=positive_int, default=4)
    parser.add_argument("--delay-ms", type=nonnegative_float, default=10,
                        help="simulated inference delay per message in milliseconds (default: 10)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dashboard", action="store_true",
                        help="enable the dashboard and wait for Enter before shutdown")
    args = parser.parse_args()
    if args.dashboard and args.mode == "sequential":
        parser.error("--dashboard requires --mode ray or both")
    return args


def report(label: str, elapsed: float, count: int) -> None:
    throughput = count / elapsed if elapsed else float("inf")
    print(f"{label:22} {elapsed:8.3f} s  {throughput:10.1f} messages/s", flush=True)


def main() -> None:
    args = parse_args()
    messages = generate_messages(args.messages, args.seed)
    delay_seconds = args.delay_ms / 1_000
    batch_count = math.ceil(len(messages) / args.batch_size)
    print(f"Messages={len(messages)}, batches={batch_count}, batch_size={args.batch_size}, "
          f"num_cpus={args.num_cpus}, delay_ms/message={args.delay_ms}, seed={args.seed}", flush=True)
    print("Concurrency demonstration using sleep; not an ML inference benchmark.", flush=True)

    sequential_results = None
    if args.mode in ("both", "sequential"):
        started = perf_counter()
        sequential_results = run_sequential(messages, args.batch_size, delay_seconds)
        sequential_seconds = perf_counter() - started
        report("Sequential execution", sequential_seconds, len(messages))

    if args.mode in ("both", "ray"):
        try:
            started = perf_counter()
            context = ray.init(address="local", num_cpus=args.num_cpus,
                               include_dashboard=args.dashboard, dashboard_host="127.0.0.1")
            startup_seconds = perf_counter() - started
            print(f"Ray initialization: {startup_seconds:.3f} s", flush=True)
            if args.dashboard:
                print(f"Dashboard: http://{context.dashboard_url}", flush=True)
                print("If prompted for a token, run: .venv/bin/ray get-auth-token | pbcopy", flush=True)
            started = perf_counter()
            ray_results = run_ray(messages, args.batch_size, delay_seconds)
            ray_seconds = perf_counter() - started
            report("Ray execution", ray_seconds, len(messages))
            report("Ray init + execution", startup_seconds + ray_seconds, len(messages))
            print(f"Flagged: {sum(result.flagged for result in ray_results)}/{len(messages)}")
            if sequential_results is not None:
                if sequential_results != ray_results:
                    raise RuntimeError("Correctness check failed: sequential and Ray results differ")
                print(f"Correctness: PASS (all {len(messages)} results match in input order)")
                print(f"Execution speedup: {sequential_seconds / ray_seconds:.2f}x")
                print(f"Including Ray init: {sequential_seconds / (startup_seconds + ray_seconds):.2f}x")
            if args.dashboard:
                try:
                    input("Inspect the dashboard; press Enter to shut down Ray. ")
                except EOFError:
                    pass
        finally:
            ray.shutdown()
    else:
        print(f"Flagged: {sum(result.flagged for result in sequential_results)}/{len(messages)}")


if __name__ == "__main__":
    main()
