"""Submit the hello script through MLPlatformClient and inspect its result."""

from pathlib import Path
from time import monotonic, sleep

from ray.job_submission import JobStatus

from mini_ml_platform import MLPlatformClient


def main() -> None:
    client = MLPlatformClient()
    working_dir = Path(__file__).resolve().parent
    job_id = client.submit_job("job_hello.py", str(working_dir))
    print(f"Submitted job: {job_id}")

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
