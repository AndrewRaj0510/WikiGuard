"""The whole pipeline in Compose: a recorded stream to a scored, labeled, API-visible edit.

Every service is the real image with its real least-privilege login, and the models are the real
files in data/runs. Only Wikimedia is a stand-in, serving recorded fixtures. Its own project name
and ports keep it apart from a dev stack, and it generates throwaway secrets, so it reads no .env.

    python scripts/compose_e2e.py                 # build, run, check, and clean up
    python scripts/compose_e2e.py --keep          # leave it running to look around
"""

import argparse
import json
import secrets
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import httpx
from argon2 import PasswordHasher

PROJECT = "wikiguard-e2e"
API_PORT = 8765
TAB = "tab-e2e-aaaaaaaa"
WAIT_READY_SECONDS = 300
WAIT_SCORED_SECONDS = 180
GRACEFUL_STOP_SECONDS = 25
# Budgets for serving images, with room to grow: going over means something heavy got in, such as
# PyTorch, which alone would add over 700 MB. Models aren't in these images.
IMAGE_BUDGET_MB = {"scorer": 550.0, "api": 550.0, "ingestor": 275.0, "archiver": 500.0}


def write_env(folder: Path) -> tuple[Path, str]:
    """Throwaway values. --env-file makes Compose read this and not the owner's .env."""
    password = secrets.token_urlsafe(24)
    user_password = secrets.token_urlsafe(24)
    hashed = PasswordHasher().hash(user_password)
    values = {
        "POSTGRES_USER": "wikiguard",
        "POSTGRES_PASSWORD": password,
        "POSTGRES_DB": "wikiguard",
        "POSTGRES_PORT": "5435",
        "REDPANDA_PORT": "19095",
        "API_PORT": str(API_PORT),
        "COMPOSE_LOGIN_PASSWORD": password,
        "USER_HASH_KEY": secrets.token_urlsafe(48),
        "SESSION_SECRET_KEY": secrets.token_urlsafe(48),
        # Single quotes, or Compose would read the hash's dollar signs as variables.
        "USER_PASSWORD_HASH": f"'{hashed}'",
        "METRICS_BEARER_TOKEN": secrets.token_urlsafe(24),
        "WIKIMEDIA_USER_AGENT": "WikiGuard-e2e/1.0 (local test; talks only to a mock)",
        "EXPLAIN_ENABLED": "false",
        "SCORER_RESCORE_SECONDS": "10",
        "LOG_LEVEL": "INFO",
    }
    path = folder / "e2e.env"
    values["APP_ENV_FILE"] = path.as_posix()
    path.write_text("".join(f"{key}={value}\n" for key, value in values.items()), "utf-8")
    return path, user_password


class Stack:
    def __init__(self, env_file: Path) -> None:
        self.base = [
            "docker",
            "compose",
            "-p",
            PROJECT,
            "--env-file",
            str(env_file),
            "-f",
            "docker-compose.yml",
            "-f",
            "docker-compose.e2e.yml",
        ]
        self.env_file = env_file

    def run(self, *args: str, check: bool = True, timeout: int = 900) -> str:
        done = subprocess.run(  # noqa: S603 - fixed commands, no shell
            [*self.base, *args],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            check=False,
        )
        if check and done.returncode != 0:
            print(done.stdout[-3000:], done.stderr[-3000:], file=sys.stderr)
            raise RuntimeError(f"docker compose {args[0]} failed")
        return done.stdout

    def clean(self) -> None:
        # Profiles have to be named, or their containers and volumes are left behind.
        self.run(
            "--profile",
            "render",
            "--profile",
            "jobs",
            "down",
            "-v",
            "--remove-orphans",
            check=False,
        )

    def psql(self, query: str) -> str:
        return self.run(
            "exec",
            "-T",
            "postgres",
            "psql",
            "-U",
            "wikiguard",
            "-d",
            "wikiguard",
            "-tA",
            "-c",
            query,
        ).strip()


def wait_for(what: str, check: Any, seconds: int) -> Any:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        result = check()
        if result:
            return result
        time.sleep(2)
    raise RuntimeError(f"timed out waiting for {what}")


def check_pipeline(stack: Stack, user_password: str) -> dict[str, Any]:
    results: dict[str, Any] = {}
    started = time.monotonic()
    client = httpx.Client(base_url=f"http://127.0.0.1:{API_PORT}", timeout=30)

    def api_ready() -> bool:
        try:
            return client.get("/ready").status_code == 200
        except httpx.HTTPError:
            return False

    wait_for("the API to be ready", api_ready, WAIT_READY_SECONDS)
    results["api_ready_seconds"] = round(time.monotonic() - started, 1)

    def scored() -> int:
        count = int(stack.psql("SELECT count(*) FROM edits WHERE latest_score IS NOT NULL"))
        total = int(stack.psql("SELECT count(*) FROM edits"))
        return count if total >= 1 and count == total and count >= 10 else 0

    results["edits_scored"] = wait_for("edits to be scored", scored, WAIT_SCORED_SECONDS)
    results["scored_by"] = stack.psql(
        "SELECT string_agg(model_name || '=' || n, ',') FROM "
        "(SELECT model_name, count(*) n FROM scores GROUP BY 1) t"
    )
    stack.run("--profile", "jobs", "run", "--rm", "labeler", timeout=300)
    results["labels"] = stack.psql(
        "SELECT string_agg(coalesce(label::text, excluded_reason) || '=' || n, ',') FROM "
        "(SELECT label, excluded_reason, count(*) n FROM labels GROUP BY 1, 2) t"
    )

    # As a browser would: sign in, read the queue, open an edit, and review it.
    response = client.post("/v1/auth/login", json={"password": user_password, "tab_id": TAB})
    response.raise_for_status()
    csrf = response.json()["csrf_token"]
    headers = {"x-tab-id": TAB, "x-csrf-token": csrf}
    queue = client.get("/v1/queue", headers=headers).json()
    results["queue_items"] = len(queue["items"])
    top = queue["items"][0]
    results["top_of_queue"] = {
        key: top[key] for key in ("score", "model_name", "used_fallback", "is_anonymous")
    }
    scores = [item["score"] for item in queue["items"]]
    results["queue_sorted"] = scores == sorted(scores, reverse=True)
    labeled_rev = stack.psql("SELECT rev_id FROM labels WHERE label IS NOT NULL LIMIT 1")
    detail = client.get(f"/v1/edits/{labeled_rev}", headers=headers).json()
    results["detail_has_label"] = detail["label"] in {0, 1}
    results["detail_has_score"] = bool(detail["scores"])
    review = client.put(f"/v1/edits/{labeled_rev}/review", headers=headers, json={"decision": "ok"})
    review.raise_for_status()
    results["review_saved"] = stack.psql("SELECT count(*) FROM reviews") == "1"
    predict = client.post("/v1/predict", headers=headers, json={"added_text": "hello world"})
    predict.raise_for_status()
    results["predict"] = {
        key: predict.json()[key] for key in ("model_name", "used_fallback", "source")
    }
    results["predict_latency_ms"] = round(predict.json()["latency_ms"], 1)
    results["leaderboard"] = [
        item["model_name"] for item in client.get("/v1/models", headers=headers).json()["items"]
    ]
    retrain = client.post("/v1/admin/retrain", headers=headers, json={"reason": "end-to-end test"})
    retrain.raise_for_status()
    stack.run("--profile", "jobs", "run", "--rm", "retrain-poller", timeout=120)
    # No trainer until Phase 5, so the row is picked up and fails with its reason.
    results["retrain_request"] = stack.psql(
        "SELECT status || ':' || (details->>'message') FROM retrain_requests"
    )

    # Graceful shutdown: SIGTERM should drain and exit well within the limit.
    began = time.monotonic()
    stack.run("stop", "-t", str(GRACEFUL_STOP_SECONDS), "api")
    results["api_stop_seconds"] = round(time.monotonic() - began, 1)
    results["api_exit_code"] = int(
        subprocess.run(  # noqa: S603 - fixed command
            ["docker", "inspect", "-f", "{{.State.ExitCode}}", f"{PROJECT}-api-1"],  # noqa: S607
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    )

    # A crash replays edits. Rewinding the offset forces that for every edit, and it must change
    # nothing: no duplicate scores, and a purged edit's text stays gone.
    fingerprint = (
        "SELECT (SELECT count(*) FROM edits), (SELECT count(*) FROM scores), "
        "(SELECT sum(score)::numeric(12,6) FROM scores)"
    )
    stack.psql(
        "UPDATE edits SET added_text = NULL, purged_at = now() WHERE rev_id = " + labeled_rev
    )
    before = stack.psql(fingerprint)
    stack.run("stop", "-t", "20", "scorer", timeout=120)
    stack.run(
        "exec",
        "-T",
        "redpanda",
        "rpk",
        "group",
        "seek",
        "scorer",
        "--to",
        "start",
        "-X",
        "brokers=redpanda:9092",
        timeout=60,
    )
    stack.run("start", "scorer", timeout=120)

    def replayed() -> bool:
        lag = stack.run(
            "exec",
            "-T",
            "redpanda",
            "rpk",
            "group",
            "describe",
            "scorer",
            "-X",
            "brokers=redpanda:9092",
            check=False,
            timeout=60,
        )
        return "TOTAL-LAG" in lag and any(
            line.startswith("TOTAL-LAG") and line.split()[-1] == "0" for line in lag.splitlines()
        )

    wait_for("the replay to finish", replayed, WAIT_SCORED_SECONDS)
    results["replay_changed_nothing"] = stack.psql(fingerprint) == before
    results["purged_text_stayed_gone"] = (
        stack.psql(f"SELECT added_text IS NULL FROM edits WHERE rev_id = {labeled_rev}") == "t"
    )

    # The archiver flushes when stopped.
    stack.run("stop", "-t", "30", "archiver")
    archived = stack.run(
        "run",
        "--rm",
        "--no-deps",
        "--entrypoint",
        "sh",
        "archiver",
        "-c",
        "find /archive -name '*.parquet' | wc -l",
        timeout=120,
    )
    results["archive_files"] = int(archived.strip().splitlines()[-1])
    results["image_mb"] = image_sizes()
    return results


def image_sizes() -> dict[str, float]:
    sizes = {}
    for service in IMAGE_BUDGET_MB:
        done = subprocess.run(  # noqa: S603 - fixed command
            ["docker", "image", "inspect", "-f", "{{.Size}}", f"{PROJECT}-{service}"],  # noqa: S607
            capture_output=True,
            text=True,
            check=True,
        )
        sizes[service] = round(int(done.stdout.strip()) / (1024 * 1024), 1)
    return sizes


def measure_render_limits(stack: Stack, user_password: str) -> dict[str, Any]:
    """The real API process and real RoBERTa at 0.1 CPU and 512 MB, one edit at a time."""
    rows = json.loads(Path("data/runs/benchmarks/sample.json").read_text("utf-8"))
    rows.sort(key=lambda row: len(row["added_text"]) + len(row["removed_text"]))
    chosen = rows[:: max(len(rows) // 15, 1)][:15] + rows[-5:]
    client = httpx.Client(base_url="http://127.0.0.1:8766", timeout=120)
    started = time.monotonic()
    stack.run("--profile", "render", "up", "-d", "--no-deps", "api-render", timeout=120)

    def ready() -> bool:
        try:
            return client.get("/ready").status_code == 200
        except httpx.HTTPError:
            return False

    wait_for("the Render-limited API", ready, WAIT_READY_SECONDS)
    results: dict[str, Any] = {"ready_seconds": round(time.monotonic() - started, 1)}
    response = client.post(
        "/v1/auth/login", json={"password": user_password, "tab_id": TAB, "take_over": True}
    )
    response.raise_for_status()
    headers = {"x-tab-id": TAB, "x-csrf-token": response.json()["csrf_token"]}
    latencies = []
    answered: dict[str, int] = {}
    for row in chosen:
        body = {
            "added_text": row["added_text"][:20_000],
            "removed_text": row["removed_text"][:20_000],
            "comment": row["comment"][:1000],
            "is_anonymous": row["is_anonymous"],
            "is_minor": row["is_minor"],
        }
        began = time.monotonic()
        reply = client.post("/v1/predict", headers=headers, json=body)
        reply.raise_for_status()
        latencies.append((time.monotonic() - began) * 1000)
        name = reply.json()["model_name"]
        answered[name] = answered.get(name, 0) + 1
    latencies.sort()
    container = f"{PROJECT}-api-render-1"
    peak = subprocess.run(  # noqa: S603 - fixed command
        ["docker", "exec", container, "cat", "/sys/fs/cgroup/memory.peak"],  # noqa: S607
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    idle_read = client.get("/v1/queue", headers=headers)
    return results | {
        "edits": len(chosen),
        "answered_by": answered,
        "request_p50_ms": round(latencies[len(latencies) // 2], 1),
        "request_max_ms": round(latencies[-1], 1),
        "peak_memory_mb": round(int(peak) / (1024 * 1024), 1),
        "queue_read_ms": round(idle_read.elapsed.total_seconds() * 1000, 1),
        "out_of_memory_killed": "true"
        in subprocess.run(  # noqa: S603 - fixed command
            ["docker", "inspect", "-f", "{{.State.OOMKilled}}", container],  # noqa: S607
            capture_output=True,
            text=True,
            check=True,
        ).stdout,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--keep", action="store_true", help="leave the stack running")
    parser.add_argument("--out", type=Path, default=Path("data/compose_e2e.json"))
    parser.add_argument(
        "--render-limits", action="store_true", help="also measure the API at 0.1 CPU and 512 MB"
    )
    args = parser.parse_args()
    folder = Path(tempfile.mkdtemp(prefix="wikiguard-e2e-"))
    env_file, user_password = write_env(folder)
    stack = Stack(env_file)
    try:
        stack.clean()
        started = time.monotonic()
        stack.run("build", timeout=1800)
        results = {"build_seconds": round(time.monotonic() - started, 1)}
        stack.run("up", "-d", timeout=600)
        results |= check_pipeline(stack, user_password)
        if args.render_limits:
            results["render_limits"] = measure_render_limits(stack, user_password)
    except Exception as error:  # noqa: BLE001 - report, with logs, and still clean up
        print(f"FAILED: {error}", file=sys.stderr)
        print(stack.run("ps", "-a", check=False), file=sys.stderr)
        print(stack.run("logs", "--tail", "40", check=False)[-8000:], file=sys.stderr)
        return 1
    finally:
        if not args.keep:
            stack.clean()
        env_file.unlink(missing_ok=True)
    args.out.write_text(json.dumps(results, indent=2) + "\n", "utf-8")
    print(json.dumps(results, indent=2))
    failed = [
        key
        for key, wanted in {
            "queue_sorted": True,
            "detail_has_label": True,
            "detail_has_score": True,
            "review_saved": True,
            "replay_changed_nothing": True,
            "purged_text_stayed_gone": True,
        }.items()
        if results.get(key) is not wanted
    ]
    if results.get("api_exit_code") not in {0}:
        failed.append("api_exit_code")
    if not results.get("archive_files"):
        failed.append("archive_files")
    for service, budget in IMAGE_BUDGET_MB.items():
        if results["image_mb"][service] > budget:
            failed.append(f"image budget: {service}")
    if failed:
        print(f"Checks failed: {failed}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
