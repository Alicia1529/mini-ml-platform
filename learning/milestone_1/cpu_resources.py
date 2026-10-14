import argparse
import os
import subprocess
from time import perf_counter, sleep

import ray


@ray.remote(num_cpus=1)
def demo_task(workload: str, seconds: float, iterations: int) -> int:
    if workload == "sleep":
        sleep(seconds)
        return 0

    total = 0
    for number in range(iterations):
        total += number * number % 97
    return total


def main() -> None:
    logical_cpus = os.cpu_count() or 1
    physical_cores = int(
        subprocess.check_output(["sysctl", "-n", "hw.physicalcpu"], text=True)
    )

    parser = argparse.ArgumentParser(description="Compare Mac CPU counts with Ray CPU resources")
    parser.add_argument("--ray-cpus", type=int, default=logical_cpus)
    parser.add_argument("--task-cpus", type=int, default=1)
    parser.add_argument("--tasks", type=int, default=4)
    parser.add_argument("--workload", choices=("sleep", "compute"), default="sleep")
    parser.add_argument("--seconds", type=float, default=1.0)
    parser.add_argument("--iterations", type=int, default=20_000_000)
    args = parser.parse_args()

    if args.ray_cpus < 1 or args.task_cpus < 1 or args.tasks < 1:
        parser.error("--ray-cpus, --task-cpus, and --tasks must be positive")
    if args.task_cpus > args.ray_cpus:
        parser.error("--task-cpus cannot exceed --ray-cpus; the tasks would wait forever")
    if args.seconds < 0 or args.iterations < 1:
        parser.error("--seconds must be non-negative and --iterations must be positive")

    print("Physical cores:", physical_cores)
    print("OS logical CPUs:", logical_cpus)

    ray.init(address="local", num_cpus=args.ray_cpus, include_dashboard=False)
    try:
        print("Ray logical CPUs:", ray.cluster_resources()["CPU"])
        print("CPU units per task:", args.task_cpus)
        print("Maximum concurrent tasks:", args.ray_cpus // args.task_cpus)

        start = perf_counter()
        refs = [
            demo_task.options(num_cpus=args.task_cpus).remote(
                args.workload, args.seconds, args.iterations
            )
            for _ in range(args.tasks)
        ]
        results = ray.get(refs)
        elapsed = perf_counter() - start
        print(f"Completed {len(results)} {args.workload} tasks in {elapsed:.2f} seconds")
    finally:
        ray.shutdown()


if __name__ == "__main__":
    main()

    # ../../.venv/bin/python cpu_resources.py --ray-cpus 2 --task-cpus 1 --tasks 4 --seconds 0.5
    # ../../.venv/bin/python cpu_resources.py --ray-cpus 2 --task-cpus 2 --tasks 4 --seconds 0.5