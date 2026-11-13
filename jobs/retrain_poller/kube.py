"""Run the trainer as a Kubernetes Job, and wait for it to finish.

The poller is the only pod with Kubernetes rights, and RBAC limits them to creating, reading, and
deleting Jobs in its own namespace (k8s/charts/wikiguard/templates/rbac.yaml). The Job's shape is
a template the Helm chart renders and mounts, so the image, resources, priority, and security
settings are reviewed in the chart, and this code only adds a name and the command.

No Kubernetes client library: three REST calls with httpx, which wikiguard_core already installs.
The token is read again before every call, because the kubelet rotates it during a long wait.
"""

import copy
import json
import os
import secrets
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx

SERVICE_ACCOUNT = Path("/var/run/secrets/kubernetes.io/serviceaccount")
POLL_SECONDS = 15
# The poller already gives a run 12 hours (TRAINER_TIMEOUT_SECONDS). The Job's own
# activeDeadlineSeconds in the template is the same, so Kubernetes stops it even if the poller dies.
LABEL = "wikiguard.dev/request"


class JobFailed(RuntimeError):
    pass


def job_name(request: dict[str, Any]) -> str:
    """Lowercase, under 63 characters, unique per attempt: trainer-rollback-12-a1b2c3."""
    kind = str(request["kind"]).replace("_", "-")
    return f"trainer-{kind}-{int(request['id'])}-{secrets.token_hex(3)}"


def build_job(template: dict[str, Any], request: dict[str, Any], args: list[str]) -> dict[str, Any]:
    """The template with this request's name, labels, and command filled in. Nothing else moves."""
    job = copy.deepcopy(template)
    name = job_name(request)
    labels = {LABEL: str(int(request["id"])), "wikiguard.dev/kind": str(request["kind"])}
    job["metadata"] = {**job.get("metadata", {}), "name": name}
    job["metadata"]["labels"] = {**job["metadata"].get("labels", {}), **labels}
    pod = job["spec"]["template"]
    pod.setdefault("metadata", {})
    pod["metadata"]["labels"] = {**pod["metadata"].get("labels", {}), **labels}
    containers = pod["spec"]["containers"]
    if len(containers) != 1:
        raise ValueError("the trainer Job template must have exactly one container")
    containers[0]["command"] = args
    containers[0].pop("args", None)
    return job


class KubernetesJobs:
    def __init__(
        self,
        template: dict[str, Any],
        client: httpx.Client,
        namespace: str,
        timeout_seconds: float,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.template = template
        self.client = client
        self.namespace = namespace
        self.timeout_seconds = timeout_seconds
        self.sleep = sleep
        self.clock = clock

    @classmethod
    def in_cluster(cls, template_path: Path, timeout_seconds: float) -> "KubernetesJobs":
        host = os.environ["KUBERNETES_SERVICE_HOST"]
        port = os.environ.get("KUBERNETES_SERVICE_PORT", "443")
        client = httpx.Client(
            base_url=f"https://{host}:{port}",
            verify=str(SERVICE_ACCOUNT / "ca.crt"),
            auth=_ServiceAccountToken(SERVICE_ACCOUNT / "token"),
            timeout=30,
        )
        namespace = (SERVICE_ACCOUNT / "namespace").read_text("utf-8").strip()
        template = json.loads(template_path.read_text("utf-8"))
        return cls(template, client, namespace, timeout_seconds)

    @property
    def _jobs(self) -> str:
        return f"/apis/batch/v1/namespaces/{self.namespace}/jobs"

    def run(self, request: dict[str, Any], args: list[str]) -> str:
        """Create the Job and block until it succeeds. Raises if it fails or runs out of time."""
        job = build_job(self.template, request, args)
        name = job["metadata"]["name"]
        self.client.post(self._jobs, json=job).raise_for_status()
        deadline = self.clock() + self.timeout_seconds
        while True:
            status = self.client.get(f"{self._jobs}/{name}").raise_for_status().json()
            outcome = finished(status)
            if outcome == "Complete":
                return name
            if outcome == "Failed":
                raise JobFailed(f"Job {name} failed: {failure_reason(status)}")
            if self.clock() >= deadline:
                self.delete(name)
                raise TimeoutError(f"Job {name} didn't finish in {self.timeout_seconds:.0f} s")
            self.sleep(POLL_SECONDS)

    def delete(self, name: str) -> None:
        # Background: the Job goes now, and Kubernetes removes its pod after.
        self.client.request(
            "DELETE",
            f"{self._jobs}/{name}",
            json={"propagationPolicy": "Background"},
        ).raise_for_status()


def finished(job: dict[str, Any]) -> str | None:
    for condition in job.get("status", {}).get("conditions") or []:
        if condition.get("status") == "True" and condition.get("type") in {"Complete", "Failed"}:
            return str(condition["type"])
    return None


def failure_reason(job: dict[str, Any]) -> str:
    for condition in job.get("status", {}).get("conditions") or []:
        if condition.get("type") == "Failed":
            return str(condition.get("reason") or "unknown")
    return "unknown"


class _ServiceAccountToken(httpx.Auth):
    def __init__(self, path: Path) -> None:
        self.path = path

    def auth_flow(self, request: httpx.Request):  # type: ignore[no-untyped-def]
        request.headers["Authorization"] = f"Bearer {self.path.read_text('utf-8').strip()}"
        yield request
