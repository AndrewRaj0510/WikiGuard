"""The cluster's security rules, checked on the rendered chart (Part 1, section 8 of the plan).

Each test renders the Helm chart the way Argo CD and the kind script do, and checks every pod:
non-root, no privilege escalation, every capability dropped, seccomp, a read-only root, limits,
no API token unless it's the poller, probes on everything long-running, default-deny networking.
A rule that's only written down gets forgotten the first time a component is added in a hurry.
"""

import json
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
CHART = ROOT / "k8s/charts/wikiguard"
pytestmark = pytest.mark.skipif(shutil.which("helm") is None, reason="needs helm")
DIGEST = "sha256:" + "a" * 64
CONFIG_SECRET_WORDS = re.compile(r"PASSWORD|SECRET|TOKEN|DSN|WEBHOOK|PING_URL|DATABASE_URL")
# Public by design: the backup's public key can only encrypt (jobs/backup).
CONFIG_ALLOWED = {"BACKUP_PUBLIC_KEY"}


class StrictLoader(yaml.SafeLoader):
    """Refuses a key that appears twice, as Kubernetes does. PyYAML would quietly keep the last.

    kubescape caught a repeated label this way (F-142), which `helm template` had rendered without
    a word.
    """


def _mapping_without_repeats(loader: StrictLoader, node: yaml.MappingNode) -> dict[Any, Any]:
    keys = [loader.construct_object(key) for key, _ in node.value]
    repeated = {key for key in keys if keys.count(key) > 1}
    if repeated:
        raise yaml.constructor.ConstructorError(
            None, None, f"repeated keys {repeated}", node.start_mark
        )
    return yaml.SafeLoader.construct_mapping(loader, node)


StrictLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _mapping_without_repeats
)


def render(*values: str, sets: tuple[str, ...] = ()) -> list[dict[str, Any]]:
    args = ["helm", "template", "wikiguard", str(CHART), "-n", "wikiguard"]
    for name in values:
        args += ["-f", str(CHART / name)]
    for value in sets:
        args += ["--set", value]
    output = subprocess.run(args, capture_output=True, text=True, check=True, encoding="utf-8")  # noqa: S603
    return [doc for doc in yaml.load_all(output.stdout, Loader=StrictLoader) if doc]  # noqa: S506


PROD = render("values.yaml", "images.yaml")
KIND = render("values-kind.yaml")


def trainer_job(docs: list[dict[str, Any]]) -> dict[str, Any]:
    config = next(d for d in docs if d["metadata"]["name"] == "wikiguard-trainer-job")
    job: dict[str, Any] = json.loads(config["data"]["trainer-job.json"])
    return job


def pods(docs: list[dict[str, Any]]) -> list[tuple[str, dict[str, Any]]]:
    """Every pod spec the chart can start, including the trainer Job the poller creates."""
    found = []
    for doc in docs:
        kind, name = doc["kind"], doc["metadata"]["name"]
        if kind in {"Deployment", "StatefulSet", "Job"}:
            found.append((f"{kind}/{name}", doc["spec"]["template"]["spec"]))
        elif kind == "CronJob":
            found.append((f"{kind}/{name}", doc["spec"]["jobTemplate"]["spec"]["template"]["spec"]))
    found.append(("Job/trainer", trainer_job(docs)["spec"]["template"]["spec"]))
    return found


def containers(spec: dict[str, Any]) -> list[dict[str, Any]]:
    return [*spec.get("initContainers", []), *spec["containers"]]


ALL_PODS = pods(PROD) + [(f"kind {name}", spec) for name, spec in pods(KIND)]


@pytest.mark.parametrize(("name", "spec"), ALL_PODS, ids=[name for name, _ in ALL_PODS])
def test_every_pod_meets_restricted_pod_security(name: str, spec: dict[str, Any]) -> None:
    pod = spec["securityContext"]
    assert pod["runAsNonRoot"] is True and pod["runAsUser"] > 0, name
    assert pod["seccompProfile"]["type"] == "RuntimeDefault", name
    for container in containers(spec):
        context = container["securityContext"]
        assert context["allowPrivilegeEscalation"] is False, (name, container["name"])
        assert context["readOnlyRootFilesystem"] is True, (name, container["name"])
        assert context["capabilities"]["drop"] == ["ALL"], (name, container["name"])
        assert "add" not in context["capabilities"], (name, container["name"])
        assert not context.get("privileged"), (name, container["name"])
    assert not spec.get("hostNetwork") and not spec.get("hostPID") and not spec.get("hostIPC")
    for volume in spec.get("volumes", []):
        assert "hostPath" not in volume, (name, volume["name"])


@pytest.mark.parametrize(("name", "spec"), ALL_PODS, ids=[name for name, _ in ALL_PODS])
def test_every_container_has_requests_and_limits(name: str, spec: dict[str, Any]) -> None:
    for container in spec["containers"]:
        resources = container.get("resources", {})
        for kind in ("requests", "limits"):
            assert {"cpu", "memory"} <= set(resources.get(kind, {})), (
                name,
                container["name"],
                kind,
            )


@pytest.mark.parametrize(("name", "spec"), ALL_PODS, ids=[name for name, _ in ALL_PODS])
def test_only_the_poller_gets_an_api_token(name: str, spec: dict[str, Any]) -> None:
    expected = name.endswith("CronJob/wikiguard-retrain-poller")
    assert spec["automountServiceAccountToken"] is expected, name


def test_long_running_pods_all_have_probes() -> None:
    for doc in PROD + KIND:
        if doc["kind"] in {"Deployment", "StatefulSet"}:
            for container in doc["spec"]["template"]["spec"]["containers"]:
                assert "livenessProbe" in container and "readinessProbe" in container, doc[
                    "metadata"
                ]["name"]


def test_batch_pods_all_have_a_deadline() -> None:
    for doc in PROD:
        if doc["kind"] == "CronJob":
            assert doc["spec"]["jobTemplate"]["spec"]["activeDeadlineSeconds"] > 0
    assert trainer_job(PROD)["spec"]["activeDeadlineSeconds"] == 12 * 3600


def test_the_scorer_outranks_the_trainer_and_training_cant_preempt() -> None:
    classes = {d["metadata"]["name"]: d for d in PROD if d["kind"] == "PriorityClass"}
    scorer = next(
        d for d in PROD if d["metadata"]["name"] == "wikiguard-scorer" and d["kind"] == "Deployment"
    )
    trainer = trainer_job(PROD)["spec"]["template"]["spec"]
    assert (
        classes[scorer["spec"]["template"]["spec"]["priorityClassName"]]["value"]
        > (classes[trainer["priorityClassName"]]["value"])
    )
    assert classes[trainer["priorityClassName"]]["preemptionPolicy"] == "Never"


def test_networking_starts_from_default_deny() -> None:
    policies = {d["metadata"]["name"]: d for d in PROD if d["kind"] == "NetworkPolicy"}
    deny = policies["default-deny"]["spec"]
    assert deny["podSelector"] == {} and set(deny["policyTypes"]) == {"Ingress", "Egress"}
    assert "ingress" not in deny and "egress" not in deny


def test_the_internet_rule_excludes_private_ranges_and_metadata() -> None:
    policy = next(d for d in PROD if d["metadata"]["name"] == "allow-internet")
    block = policy["spec"]["egress"][0]["to"][0]["ipBlock"]
    assert block["cidr"] == "0.0.0.0/0"
    assert {
        "10.0.0.0/8",
        "172.16.0.0/12",
        "192.168.0.0/16",
        "169.254.0.0/16",
        "100.64.0.0/10",
    } <= set(block["except"])
    assert [port["port"] for port in policy["spec"]["egress"][0]["ports"]] == [443]


def test_only_listed_components_reach_redpanda() -> None:
    clients = set()
    for doc in PROD:
        template = doc.get("spec", {}).get("template") or doc.get("spec", {}).get(
            "jobTemplate", {}
        ).get("spec", {}).get("template")
        if template and template["metadata"]["labels"].get("wikiguard.dev/redpanda") == "true":
            clients.add(doc["metadata"]["name"])
    assert clients == {
        "wikiguard-ingestor",
        "wikiguard-scorer",
        "wikiguard-archiver",
        "wikiguard-labeler",
    }


def test_the_poller_may_only_manage_jobs_in_its_namespace() -> None:
    role = next(d for d in PROD if d["kind"] == "Role")
    assert role["rules"] == [
        {"apiGroups": ["batch"], "resources": ["jobs"], "verbs": ["create", "get", "delete"]}
    ]
    assert not [d for d in PROD if d["kind"] in {"ClusterRole", "ClusterRoleBinding"}]
    binding = next(d for d in PROD if d["kind"] == "RoleBinding")
    assert [s["name"] for s in binding["subjects"]] == ["wikiguard-retrain-poller"]


def test_no_secret_is_rendered_and_config_holds_none() -> None:
    assert not [d for d in PROD + KIND if d["kind"] == "Secret"]
    config = next(d for d in PROD if d["metadata"]["name"] == "wikiguard-config")["data"]
    for key in config:
        assert key in CONFIG_ALLOWED or not CONFIG_SECRET_WORDS.search(key), key


def test_images_come_by_digest_once_cd_records_one() -> None:
    docs = render("values.yaml", "images.yaml", sets=(f"digests.scorer={DIGEST}",))
    scorer = next(
        d for d in docs if d["metadata"]["name"] == "wikiguard-scorer" and d["kind"] == "Deployment"
    )
    image = scorer["spec"]["template"]["spec"]["containers"][0]["image"]
    assert image == f"ghcr.io/andrewraj0510/wikiguard-scorer@{DIGEST}"


def test_every_image_the_chart_runs_is_one_cd_publishes_and_pins() -> None:
    workflow = yaml.safe_load((ROOT / ".github/workflows/cd.yml").read_text("utf-8"))
    published = {row["name"] for row in workflow["jobs"]["build"]["strategy"]["matrix"]["image"]}
    script = next(
        step["run"]
        for step in workflow["jobs"]["cluster-images"]["steps"]
        if "digests:" in step.get("run", "")
    )
    pinned = set(
        re.search(r"for name in (.*?); do", script, re.DOTALL).group(1).replace("\\", " ").split()
    )  # type: ignore[union-attr]
    used = set()
    for _, spec in pods(PROD):
        for container in containers(spec):
            match = re.match(r"ghcr\.io/andrewraj0510/wikiguard-([a-z-]+)[:@]", container["image"])
            if match:
                used.add(match.group(1))
    assert used <= published
    assert used == pinned


def test_the_trainer_job_template_has_one_container_for_the_poller() -> None:
    spec = trainer_job(PROD)["spec"]
    assert spec["backoffLimit"] == 0
    assert len(spec["template"]["spec"]["containers"]) == 1
    assert spec["template"]["metadata"]["labels"]["wikiguard.dev/database"] == "true"


def test_every_dashboard_is_current_and_reads_only_what_the_observer_may() -> None:
    result = subprocess.run(  # noqa: S603
        [sys.executable, str(ROOT / "scripts/build_dashboards.py"), "--check"],
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    migration = (
        ROOT / "jobs/migrate/migrations/versions/0005_backup_and_observer_roles.py"
    ).read_text("utf-8")
    listed = re.search(r"OBSERVER_TABLES = \((.*?)\)", migration, re.DOTALL)
    assert listed is not None
    allowed = set(re.findall(r'"(\w+)"', listed.group(1)))
    for path in (CHART / "dashboards").glob("*.json"):
        for panel in json.loads(path.read_text("utf-8"))["panels"]:
            for target in panel.get("targets", []):
                for table in re.findall(r"\bFROM\s+(\w+)", target.get("rawSql", ""), re.IGNORECASE):
                    assert table in allowed, (path.name, table)


def test_kind_runs_no_job_that_needs_the_owners_keys() -> None:
    suspended = {
        d["metadata"]["name"] for d in KIND if d["kind"] == "CronJob" and d["spec"].get("suspend")
    }
    assert suspended == {"wikiguard-purge-snapshots", "wikiguard-purge-snapshots-full"}
    assert "initContainers" not in trainer_job(KIND)["spec"]["template"]["spec"]
