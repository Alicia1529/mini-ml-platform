"""Submit the moderation script through MLPlatformClient and inspect its result."""

import argparse
from pathlib import Path
from time import monotonic, sleep

from ray.job_submission import JobStatus

from mini_ml_platform.client import MLPlatformClient


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare Job driver CPU reservations.")
    parser.add_argument("--entrypoint-cpus", type=int, choices=(0, 1), default=0)
    args = parser.parse_args()

    client = MLPlatformClient()
    working_dir = Path(__file__).resolve().parent
    job_id = client.submit_job(
        "job_moderation.py", str(working_dir), entrypoint_num_cpus=args.entrypoint_cpus
    )
    print(f"Submitted job: {job_id} (driver CPU reservation: {args.entrypoint_cpus})")

    deadline = monotonic() + 60
    while True:
        status = client.get_status(job_id)
        print(f"Status: {status}")
        if status in {JobStatus.SUCCEEDED, JobStatus.FAILED, JobStatus.STOPPED}:
            break
        if monotonic() >= deadline:
            raise TimeoutError(f"Job {job_id} did not finish within 60 seconds")
        sleep(1)

    print(client.get_logs(job_id), end="")
    if status != JobStatus.SUCCEEDED:
        raise RuntimeError(f"Job {job_id} finished with status {status}")


if __name__ == "__main__":
    main()
