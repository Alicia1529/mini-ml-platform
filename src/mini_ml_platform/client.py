"""Small wrapper around Ray's Jobs submission API."""

from shlex import quote

from ray.job_submission import JobStatus, JobSubmissionClient


class MLPlatformClient:
    def __init__(self, address: str = "http://127.0.0.1:8265"):
        self._jobs = JobSubmissionClient(address)

    def submit_job(
        self,
        script: str,
        working_dir: str,
        submission_id: str | None = None,
        entrypoint_num_cpus: float | None = None,
    ) -> str:
        """Submit a Python script and return its Ray submission ID."""
        return self._jobs.submit_job(
            entrypoint=f"python {quote(script)}",
            runtime_env={"working_dir": working_dir},
            submission_id=submission_id,
            entrypoint_num_cpus=entrypoint_num_cpus,
        )

    def stop_job(self, job_id: str) -> bool:
        """Request that a running job stop."""
        return self._jobs.stop_job(job_id)

    def get_status(self, job_id: str) -> JobStatus:
        """Return the job's current status."""
        return self._jobs.get_job_status(job_id)

    def get_logs(self, job_id: str) -> str:
        """Return logs produced by the job so far."""
        return self._jobs.get_job_logs(job_id)
