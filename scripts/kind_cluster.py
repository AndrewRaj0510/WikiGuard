"""Run WikiGuard's cluster half on kind, on the laptop: the same chart the VM runs.

    python scripts/kind_cluster.py up                  # everything below, in order
    python scripts/kind_cluster.py images [NAME ...]   # build and load images into the node
    python scripts/kind_cluster.py monitoring          # kube-prometheus-stack and the alert sink
    python scripts/kind_cluster.py install             # throwaway secrets, then the chart
    python scripts/kind_cluster.py down                # delete the cluster, its volume, its images

Disk space: once an image is loaded into the node, the host's copy is deleted, and the unused
build cache is pruned (scripts/docker_space.py), since the node keeps its own copy. `down` also
deletes the node's volume, which holds every loaded image (16.5 GB measured, 2026-11-13). Pass
`--keep-images` to keep the host's copies and the cache, for faster rebuilds.

What's real: every image, the chart and its NetworkPolicies, Pod Security, RBAC, Redpanda,
Prometheus, Alertmanager, Grafana, the dashboards, and the live Wikimedia stream. What stands in:
Postgres runs in the cluster instead of Supabase, the archive and backups stay on volumes with no
bucket, the models are read from DagsHub's public MLflow, and an in-cluster sink plays Discord and
Healthchecks.io. Secrets are generated here, saved to data/kind/ (git-ignored), and sent to kubectl
on stdin, never on a command line. It never reads .env.
"""

import argparse
import base64
import json
import secrets
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import docker_space  # noqa: E402 - a sibling script

ROOT = Path(__file__).resolve().parents[1]
CLUSTER = "wikiguard"
CONTEXT = f"kind-{CLUSTER}"
NAMESPACE = "wikiguard"
REGISTRY = "kind.local"
TAG = "dev"
STATE = ROOT / "data" / "kind"
CHART = ROOT / "k8s" / "charts" / "wikiguard"
MONITORING_CHART = ("prometheus-community/kube-prometheus-stack", "93.2.2")
# Render runs the API, and the snapshot purge needs the owner's DagsHub key, so kind skips both.
NOT_ON_KIND = {"api", "purge-snapshots"}
LOGINS = ("scorer", "labeler", "trainer", "maintenance", "backup", "observer")


def run(*args: str, stdin: str | None = None, check: bool = True, quiet: bool = False) -> str:
    result = subprocess.run(  # noqa: S603 - fixed tools, arguments from this file
        list(args),
        input=stdin,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
        cwd=ROOT,
    )
    if check and result.returncode != 0:
        sys.stderr.write(result.stdout[-2000:] + result.stderr[-4000:])
        raise SystemExit(f"failed: {' '.join(args[:4])} ...")
    if not quiet and result.stdout.strip():
        print(result.stdout.strip()[-1500:])  # noqa: T201
    return result.stdout


def kubectl(*args: str, stdin: str | None = None, check: bool = True, quiet: bool = False) -> str:
    return run("kubectl", "--context", CONTEXT, *args, stdin=stdin, check=check, quiet=quiet)


def images_from_cd() -> dict[str, str]:
    """The same list CD publishes, so kind can't drift from production."""
    workflow = yaml.safe_load((ROOT / ".github/workflows/cd.yml").read_text("utf-8"))
    rows = workflow["jobs"]["build"]["strategy"]["matrix"]["image"]
    return {row["name"]: row["dockerfile"] for row in rows if row["name"] not in NOT_ON_KIND}


def build_images(names: list[str], keep_images: bool = False) -> None:
    every = images_from_cd()
    for name in names or list(every):
        tag = f"{REGISTRY}/wikiguard-{name}:{TAG}"
        started = time.monotonic()
        run("docker", "build", "-f", every[name], "-t", tag, ".", quiet=True)
        run("kind", "load", "docker-image", tag, "--name", CLUSTER, quiet=True)
        # The node has its own copy now, so the host's is only taking space.
        if not keep_images:
            docker_space.remove_images([tag])
        print(f"{name}: built and loaded in {time.monotonic() - started:.0f} s")  # noqa: T201
    if not keep_images:
        docker_space.report(
            "host images removed after loading", [], docker_space.prune_build_cache()
        )


def delete_cluster(keep_images: bool = False) -> None:
    """The node's volume holds every image loaded into it, so it goes with the cluster."""
    nodes = run("kind", "get", "nodes", "--name", CLUSTER, check=False, quiet=True).split()
    volumes = [volume for node in nodes for volume in docker_space.container_volumes(node)]
    run("kind", "delete", "cluster", "--name", CLUSTER)
    removed = docker_space.remove_volumes(volumes)
    images: list[str] = []
    cache = ""
    if not keep_images:
        images = docker_space.remove_images(docker_space.images_matching(f"{REGISTRY}/wikiguard-*"))
        cache = docker_space.prune_build_cache()
    docker_space.report("volumes and images removed", removed + images, cache)


def saved_secrets() -> dict[str, str]:
    """Generated once and kept, because the Postgres volume keeps the first password it saw."""
    STATE.mkdir(parents=True, exist_ok=True)
    path = STATE / "secrets.json"
    if path.is_file():
        values: dict[str, str] = json.loads(path.read_text("utf-8"))
        return values
    sys.path.insert(0, str(ROOT / "jobs" / "backup"))
    from backup import keygen

    public, private = keygen()
    values = {
        "postgres": secrets.token_urlsafe(24),
        "logins": secrets.token_urlsafe(24),
        "user_hash_key": secrets.token_urlsafe(48),
        "grafana": secrets.token_urlsafe(18),
        # A throwaway backup key pair: the public half goes to the cluster, the private half stays
        # here for the restore rehearsal (docs/runbooks/backup-restore.md).
        "backup_public_key": public,
    }
    path.write_text(json.dumps(values, indent=2), "utf-8")
    (STATE / "grafana_password.txt").write_text(values["grafana"], "utf-8")
    (STATE / "backup_private_key.txt").write_text(private, "utf-8")
    return values


def secret(name: str, namespace: str, data: dict[str, str]) -> dict[str, Any]:
    return {
        "apiVersion": "v1",
        "kind": "Secret",
        "metadata": {"name": name, "namespace": namespace},
        "type": "Opaque",
        "data": {key: base64.b64encode(value.encode()).decode() for key, value in data.items()},
    }


def apply(documents: list[dict[str, Any]]) -> None:
    kubectl("apply", "-f", "-", stdin=yaml.safe_dump_all(documents), quiet=True)


def database_url(login: str, password: str) -> str:
    return f"postgresql://{login}_login:{password}@postgres:5432/wikiguard"


def app_secrets(values: dict[str, str]) -> list[dict[str, Any]]:
    login = values["logins"]
    per_component = {
        "postgres": {"POSTGRES_PASSWORD": values["postgres"]},
        "migrate": {
            "DATABASE_MIGRATIONS_URL": f"postgresql://wikiguard:{values['postgres']}@postgres:5432/wikiguard",
            "COMPOSE_LOGIN_PASSWORD": login,
        },
        "ingestor": {"USER_HASH_KEY": values["user_hash_key"]},
        "scorer": {"DATABASE_URL": database_url("scorer", login)},
        "archiver": {},
        "labeler": {"DATABASE_URL": database_url("labeler", login)},
        "monitor": {"DATABASE_URL": database_url("trainer", login)},
        "purge": {
            "DATABASE_URL": database_url("maintenance", login),
            "USER_HASH_KEY": values["user_hash_key"],
        },
        "purge-snapshots": {},
        "retention": {"DATABASE_URL": database_url("maintenance", login)},
        "backup": {"DATABASE_URL": database_url("backup", login)},
        "retrain-poller": {"DATABASE_URL": database_url("trainer", login)},
        "trainer": {"DATABASE_URL": database_url("trainer", login)},
    }
    return [secret(f"wikiguard-{name}", NAMESPACE, data) for name, data in per_component.items()]


def monitoring_secrets(values: dict[str, str]) -> list[dict[str, Any]]:
    sink = "http://alert-sink.monitoring.svc:8080"
    return [
        secret(
            "grafana-admin",
            "monitoring",
            {"admin-user": "admin", "admin-password": values["grafana"]},
        ),
        secret(
            "wikiguard-alerting",
            "monitoring",
            {"discord-webhook-url": f"{sink}/discord", "healthchecks-url": f"{sink}/healthchecks"},
        ),
        secret(
            "grafana-observer-db",
            "monitoring",
            {
                "OBSERVER_DB_HOST": "postgres.wikiguard.svc:5432",
                "OBSERVER_DB_NAME": "wikiguard",
                "OBSERVER_DB_USER": "observer_login",
                "OBSERVER_DB_PASSWORD": values["logins"],
                "OBSERVER_DB_SSLMODE": "disable",
            },
        ),
    ]


def create_cluster() -> None:
    clusters = run("kind", "get", "clusters", quiet=True).split()
    if CLUSTER not in clusters:
        run("kind", "create", "cluster", "--config", "k8s/kind/cluster.yaml", "--wait", "120s")
    kubectl("apply", "-f", "k8s/platform/namespaces.yaml", quiet=True)


def install_monitoring() -> None:
    apply(monitoring_secrets(saved_secrets()))
    kubectl("apply", "-f", "k8s/kind/alert-sink.yaml", quiet=True)
    chart, version = MONITORING_CHART
    run(
        "helm", "upgrade", "--install", "kps", chart, "--version", version,
        "--kube-context", CONTEXT, "-n", "monitoring",
        "-f", "k8s/platform/monitoring-values.yaml", "-f", "k8s/kind/monitoring-values.yaml",
        "--wait", "--timeout", "15m", quiet=True,
    )  # fmt: skip
    print("monitoring installed")  # noqa: T201


def api_server_cidr() -> str:
    """Where the API server really answers. A NetworkPolicy sees the address after the Service."""
    address = kubectl(
        "get", "endpointslices", "-n", "default", "-l", "kubernetes.io/service-name=kubernetes",
        "-o", "jsonpath={.items[0].endpoints[0].addresses[0]}", quiet=True,
    ).strip()  # fmt: skip
    return f"{address}/32"


def install_chart(extra: list[str]) -> None:
    values = saved_secrets()
    apply(app_secrets(values))
    run(
        "helm", "upgrade", "--install", "wikiguard", str(CHART),
        "--kube-context", CONTEXT, "-n", NAMESPACE,
        "-f", str(CHART / "values-kind.yaml"),
        "--set", f"network.kubeApi.cidr={api_server_cidr()}",
        "--set", f"config.BACKUP_PUBLIC_KEY={values.get('backup_public_key', '')}",
        *extra, "--timeout", "15m", quiet=True,
    )  # fmt: skip
    for kind, name in (
        ("statefulset", "redpanda"),
        ("deployment", "postgres"),
        ("deployment", "wikiguard-ingestor"),
        ("deployment", "wikiguard-archiver"),
        ("deployment", "wikiguard-scorer"),
    ):
        kubectl("rollout", "status", f"{kind}/{name}", "-n", NAMESPACE, "--timeout", "20m")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=["up", "images", "monitoring", "install", "down"])
    parser.add_argument("names", nargs="*", help="images to build, for `images`")
    parser.add_argument("--set", action="append", default=[], help="extra helm --set, for install")
    parser.add_argument(
        "--keep-images",
        action="store_true",
        help="keep the host's image copies and the build cache (faster rebuilds, more disk)",
    )
    args = parser.parse_args()
    extra = [part for value in args.set for part in ("--set", value)]
    if args.command == "down":
        delete_cluster(args.keep_images)
        return 0
    if args.command in {"up"}:
        create_cluster()
    if args.command in {"up", "images"}:
        build_images(args.names, args.keep_images)
    if args.command in {"up", "monitoring"}:
        install_monitoring()
    if args.command in {"up", "install"}:
        install_chart(extra)
    return 0


if __name__ == "__main__":
    sys.exit(main())
