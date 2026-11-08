"""Check the Ray Jobs SDK wrapper without starting a cluster."""

from unittest.mock import Mock, call

import pytest
from ray.job_submission import JobStatus, JobSubmissionClient

from mini_ml_platform import MLPlatformClient, client as client_module


def test_client_is_available_at_package_root():
    assert MLPlatformClient is client_module.MLPlatformClient


@pytest.fixture
def client_and_jobs(monkeypatch):
    jobs = Mock(spec=JobSubmissionClient)
    factory = Mock(return_value=jobs)
    monkeypatch.setattr(client_module, "JobSubmissionClient", factory)

    client = client_module.MLPlatformClient("http://ray.example:8265")
    factory.assert_called_once_with("http://ray.example:8265")
    return client, jobs


def test_submit_job_passes_entrypoint_and_working_dir(client_and_jobs):
    client, jobs = client_and_jobs
    submission_id = "test_submission_456"
    jobs.submit_job.return_value = submission_id

    job_id = client.submit_job(
        "scripts/job moderation.py", "/tmp/project",
        submission_id=submission_id, entrypoint_num_cpus=1,
    )

    assert job_id == submission_id
    jobs.submit_job.assert_called_once_with(
        entrypoint="python 'scripts/job moderation.py'",
        runtime_env={"working_dir": "/tmp/project"},
        submission_id=submission_id,
        entrypoint_num_cpus=1,
    )


def test_submit_job_uses_generated_id_by_default(client_and_jobs):
    client, jobs = client_and_jobs
    jobs.submit_job.return_value = "raysubmit_123"

    assert client.submit_job("job_hello.py", "/tmp/project") == "raysubmit_123"
    jobs.submit_job.assert_called_once_with(
        entrypoint="python job_hello.py",
        runtime_env={"working_dir": "/tmp/project"},
        submission_id=None,
        entrypoint_num_cpus=None,
    )


def test_job_status_logs_and_stop_result_are_returned(client_and_jobs):
    client, jobs = client_and_jobs
    job_id = "raysubmit_123"
    jobs.get_job_status.return_value = JobStatus.RUNNING
    jobs.get_job_logs.return_value = "started\n"
    jobs.stop_job.side_effect = [True, False]

    assert client.get_status(job_id) == JobStatus.RUNNING
    assert client.get_logs(job_id) == "started\n"
    assert client.stop_job(job_id) is True
    assert client.stop_job(job_id) is False

    jobs.get_job_status.assert_called_once_with(job_id)
    jobs.get_job_logs.assert_called_once_with(job_id)
    assert jobs.stop_job.call_args_list == [call(job_id), call(job_id)]
