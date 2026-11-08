"""Smoke-test the running Ray Serve HTTP endpoint."""

import json
import os
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest


@pytest.fixture(scope="module")
def serve_url():
    url = os.environ.get("SERVE_URL")
    if not url:
        pytest.skip("Start Ray Serve and set SERVE_URL to run HTTP tests")
    return url


@pytest.mark.serve
@pytest.mark.parametrize(
    ("payload", "expected_status", "expected_body"),
    [
        ('{"text":"SCAM"}', 200, {"flagged": True, "matched_keywords": ["scam"]}),
        ('{"text":"hello"}', 200, {"flagged": False, "matched_keywords": []}),
        ('{"text":123}', 400, {"error": "text must be a string"}),
        ('{}', 400, {"error": "text must be a string"}),
        ('{"text":', 400, {"error": "invalid JSON"}),
    ],
)
def test_moderation_http(serve_url, payload, expected_status, expected_body):
    request = Request(
        serve_url,
        data=payload.encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        response = urlopen(request, timeout=10)
    except HTTPError as error:
        response = error

    with response:
        assert response.status == expected_status
        body = json.load(response)
    for key, value in expected_body.items():
        assert body[key] == value
