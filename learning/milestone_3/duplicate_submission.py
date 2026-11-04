"""Show that Ray rejects a second job with the same submission ID."""

from pathlib import Path
from uuid import uuid4

from mini_ml_platform.client import MLPlatformClient


def main() -> None:
    client = MLPlatformClient()
    working_dir = Path(__file__).resolve().parent

    submission_id = f"demo-{uuid4().hex}"
    first = client.submit_job("job_hello.py", str(working_dir), submission_id=submission_id)
    print(f"Submitted job: {first}, submission_id: {submission_id}")

    try:
        client.submit_job(
            "job_hello.py", str(working_dir), submission_id=submission_id
        )
    except RuntimeError as exc:
        print(f"Second submission failed: {exc}")


if __name__ == "__main__":
    main()
