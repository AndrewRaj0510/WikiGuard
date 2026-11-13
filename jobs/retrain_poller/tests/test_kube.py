"""The Kubernetes launcher against a fake API server: what it sends, and how it reads the answer.

A real cluster ran it too: kind, through the chart (docs/metrics.md, section 19).
"""

import json
from typing import Any

import httpx
import pytest
from kube import JobFailed, KubernetesJobs, build_job, finished, job_name
from poller import job_launcher

TEMPLATE: dict[str, Any] = {
    "apiVersion": "batch/v1",
    "kind": "Job",
    "metadata": {"labels": {"app.kubernetes.io/name": "trainer"}},
    "spec": {
        "backoffLimit": 0,
        "activeDeadlineSeconds": 43200,
        "template": {
            "metadata": {"labels": {"app.kubernetes.io/name": "trainer"}},
            "spec": {
                "restartPolicy": "Never",
                "containers": [{"name": "trainer", "image": "trainer@sha256:abc", "args": ["x"]}],
            },
        },
    },
}
REQUEST = {"id": 12, "kind": "shadow_check", "run_id": 4}


class FakeApi:
    """Answers like the API server: the Job is created, then reports `states` in turn."""

    def __init__(self, states: list[str | None]) -> None:
        self.states = states
        self.calls: list[tuple[str, str, Any]] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content) if request.content else None
        self.calls.append((request.method, request.url.path, body))
        if request.method == "POST":
            return httpx.Response(201, json=body)
        if request.method == "DELETE":
            return httpx.Response(200, json={})
        state = self.states.pop(0) if self.states else None
        conditions = [{"type": state, "status": "True", "reason": "BackoffLimitExceeded"}]
        return httpx.Response(200, json={"status": {"conditions": conditions if state else []}})


def launcher(api: FakeApi, timeout: float = 3600) -> tuple[KubernetesJobs, list[float]]:
    slept: list[float] = []
    now = [0.0]

    def sleep(seconds: float) -> None:
        slept.append(seconds)
        now[0] += seconds

    client = httpx.Client(base_url="https://k8s.test", transport=httpx.MockTransport(api.handler))
    jobs = KubernetesJobs(TEMPLATE, client, "wikiguard", timeout, sleep=sleep, clock=lambda: now[0])
    return jobs, slept


def test_the_job_gets_the_command_and_a_name_and_nothing_else_moves() -> None:
    job = build_job(TEMPLATE, REQUEST, ["python", "-m", "trainer.lifecycle", "shadow-check"])
    container = job["spec"]["template"]["spec"]["containers"][0]
    assert container["command"] == ["python", "-m", "trainer.lifecycle", "shadow-check"]
    assert "args" not in container
    assert container["image"] == "trainer@sha256:abc"
    assert job["spec"]["activeDeadlineSeconds"] == 43200
    assert job["metadata"]["name"].startswith("trainer-shadow-check-12-")
    assert job["metadata"]["labels"]["wikiguard.dev/request"] == "12"
    assert job["spec"]["template"]["metadata"]["labels"]["app.kubernetes.io/name"] == "trainer"
    # The template itself is left as it was, for the next request.
    assert TEMPLATE["spec"]["template"]["spec"]["containers"][0]["args"] == ["x"]


def test_names_are_valid_and_differ_between_attempts() -> None:
    names = {job_name(REQUEST) for _ in range(20)}
    assert len(names) == 20
    assert all(len(name) <= 63 and name == name.lower() and "_" not in name for name in names)


def test_a_template_with_two_containers_is_refused() -> None:
    broken = json.loads(json.dumps(TEMPLATE))
    broken["spec"]["template"]["spec"]["containers"].append({"name": "extra"})
    with pytest.raises(ValueError, match="exactly one container"):
        build_job(broken, REQUEST, ["x"])


def test_waits_until_the_job_completes() -> None:
    api = FakeApi([None, None, "Complete"])
    jobs, slept = launcher(api)
    name = jobs.run(REQUEST, ["python", "-m", "trainer.lifecycle", "shadow-check"])
    assert api.calls[0][:2] == ("POST", "/apis/batch/v1/namespaces/wikiguard/jobs")
    assert api.calls[-1][1].endswith(f"/jobs/{name}")
    assert slept == [15, 15]


def test_a_failed_job_raises_with_its_reason() -> None:
    jobs, _ = launcher(FakeApi(["Failed"]))
    with pytest.raises(JobFailed, match="BackoffLimitExceeded"):
        jobs.run(REQUEST, ["x"])


def test_a_job_past_the_limit_is_deleted() -> None:
    api = FakeApi([None] * 100)
    jobs, _ = launcher(api, timeout=60)
    with pytest.raises(TimeoutError):
        jobs.run(REQUEST, ["x"])
    method, _, body = api.calls[-1]
    assert method == "DELETE" and body == {"propagationPolicy": "Background"}


def test_finished_ignores_conditions_that_arent_true() -> None:
    assert finished({"status": {"conditions": [{"type": "Failed", "status": "False"}]}}) is None
    assert finished({"status": {}}) is None
    assert finished({}) is None


def test_the_poller_passes_only_the_kind_and_numbers_to_the_job() -> None:
    api = FakeApi(["Complete"])
    jobs, _ = launcher(api)
    job_launcher("python -m trainer.lifecycle", jobs)(REQUEST)
    command = api.calls[0][2]["spec"]["template"]["spec"]["containers"][0]["command"]
    assert command == [
        "python", "-m", "trainer.lifecycle", "shadow-check", "--request", "12", "--run", "4",
    ]  # fmt: skip
