"""Submit a slow job, stop it, and inspect its final status."""

from pathlib import Path
from time import monotonic, sleep

from ray.job_submission import JobStatus

from mini_ml_platform.client import MLPlatformClient


def main() -> None:
    client = MLPlatformClient()
    working_dir = Path(__file__).resolve().parent
    job_id = client.submit_job("job_slow.py", str(working_dir))
    print(f"Submitted job: {job_id}")

    deadline = monotonic() + 60
    stop_time = monotonic() + 15
    stop_requested = False
    while True:
        status = client.get_status(job_id)
        print(f"Status: {status}")
        if status in {JobStatus.SUCCEEDED, JobStatus.FAILED, JobStatus.STOPPED}:
            break
        if not stop_requested and monotonic() >= stop_time:
            print(f"Stopping job {job_id} after 15 seconds...")
            stop_requested = client.stop_job(job_id)
            print(f"Stop request accepted: {stop_requested}")
        if monotonic() >= deadline:
            raise TimeoutError(f"Job {job_id} did not finish within 60 seconds")
        sleep(1)

    print(client.get_logs(job_id), end="")
    if status != JobStatus.STOPPED:
        raise RuntimeError(f"Job {job_id} finished with status {status}, expected STOPPED")


if __name__ == "__main__":
    main()
