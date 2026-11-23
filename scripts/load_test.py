"""Run the Phase 8 load tests and record what they measure (docs/performance.md).

    python scripts/load_test.py up                    # build, load the demo's data, start the API
    python scripts/load_test.py predict --label before
    python scripts/load_test.py reads --mode normal   # also: spike, soak (an hour)
    python scripts/load_test.py explain
    python scripts/load_test.py replay                # edits.raw at 5x the real rate
    python scripts/load_test.py down                  # stop it, delete its volumes and images

What's under test is what the public demo runs (DEC-091): the API at Render's free limits (0.1 CPU,
512 MB, `api-render`) with render.yaml's settings, in snapshot mode, over the demo's dataset loaded
by scripts/load_demo.py, which is 25,294 real edits. What differs from Render: Postgres is local,
not Supabase across the internet, so reads miss Supabase's network time, and the CPU limit is
Docker Desktop's, not Render's (F-105). The replay runs the real scorer at the production chart's
limit (1.5 CPU, 3 GiB) with all five models.

k6 drives the HTTP tests (loadtest/*.js). Each run writes data/load_test/SCENARIO-LABEL.json: the k6
summary's numbers, the API container's peak memory and CPU, and the API's own request histogram.
Nothing calls Wikimedia or an LLM: Explain talks to tests/e2e/fake_llm.py. The stack reads no .env.
"""

import argparse
import datetime as dt
import json
import os
import re
import secrets
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent))
import compose_e2e as e2e  # noqa: E402 - shares its Stack and settings

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / "data" / "load_test"
STATE = WORK / "state.json"
API = "http://127.0.0.1:8766"
SNAPSHOT = "snapshots/snapshot-20261104-28c825af4b19"
SAMPLE = ROOT / "data/runs/benchmarks/sample.json"
API_CONTAINER = f"{e2e.PROJECT}-api-render-1"
SCORER_CONTAINER = f"{e2e.PROJECT}-scorer-1"
REPLAY_OFFSET = 50_000_000_000


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)  # noqa: T201


def k6_binary() -> str:
    found = shutil.which("k6")
    if found:
        return found
    default = Path("C:/Program Files/k6/k6.exe")
    if default.exists():
        return str(default)
    raise SystemExit("k6 isn't installed: winget install GrafanaLabs.k6")


class Stack(e2e.Stack):
    def __init__(self, env_file: Path) -> None:
        super().__init__(env_file)
        self.base += ["-f", "docker-compose.loadtest.yml"]


def env_values(path: Path) -> dict[str, str]:
    values = {}
    for line in path.read_text("utf-8").splitlines():
        key, _, value = line.partition("=")
        values[key] = value.strip("'")
    return values


def make_certificate(folder: Path) -> None:
    """A throwaway CA-signed-by-itself certificate for the fake LLM's two host names."""
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import ec
    from cryptography.x509.oid import NameOID

    key = ec.generate_private_key(ec.SECP256R1())
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "WikiGuard load-test fake LLM")])
    now = dt.datetime.now(dt.UTC)
    hosts = ["api.openai.com", "generativelanguage.googleapis.com"]
    certificate = (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(name)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - dt.timedelta(minutes=5))
        .not_valid_after(now + dt.timedelta(days=2))
        .add_extension(x509.SubjectAlternativeName([x509.DNSName(h) for h in hosts]), False)
        .add_extension(x509.BasicConstraints(ca=True, path_length=None), True)
        .sign(key, hashes.SHA256())
    )
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "cert.pem").write_bytes(certificate.public_bytes(serialization.Encoding.PEM))
    (folder / "key.pem").write_bytes(
        key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        )
    )


def load_state() -> dict[str, Any]:
    if not STATE.exists():
        raise SystemExit("No stack: run `python scripts/load_test.py up` first")
    state: dict[str, Any] = json.loads(STATE.read_text("utf-8"))
    os.environ.update(state["compose_env"])
    return state


def edge() -> dict[str, str]:
    """With EDGE_SHARED_SECRET set, the API refuses requests that don't come through the Worker."""
    return {"x-edge-secret": os.environ["LT_EDGE_SECRET"], "x-wikiguard-client-ip": "10.255.0.1"}


def wait_ready(seconds: float = 600) -> float:
    started = time.monotonic()
    while time.monotonic() - started < seconds:
        try:
            if httpx.get(f"{API}/ready", headers=edge(), timeout=5).status_code == 200:
                return round(time.monotonic() - started, 1)
        except httpx.HTTPError:
            pass
        time.sleep(2)
    raise RuntimeError("the API never became ready")


def start_api(stack: Stack, **switches: str) -> float:
    for key in ("LT_EXPLAIN_ENABLED", "LT_PREDICT_TIMEOUT_MS"):
        os.environ.pop(key, None)
    os.environ.update(switches)
    stack.run(
        "--profile", "render", "up", "-d", "--no-deps", "--force-recreate",
        "api-render", "fake-llm", timeout=600,
    )  # fmt: skip
    return wait_ready()


def up(args: argparse.Namespace) -> None:
    WORK.mkdir(parents=True, exist_ok=True)
    env_file, password = e2e.write_env(WORK)
    e2e.write_private(WORK / "user_password.txt", password)
    make_certificate(WORK / "cert")
    compose_env = {
        "LT_EDGE_SECRET": secrets.token_urlsafe(32),
        "LT_CERT_DIR": (WORK / "cert").resolve().as_posix(),
    }
    os.environ.update(compose_env)
    stack = Stack(env_file)
    stack.clean()
    log("building the images and migrating")
    stack.run("build", "migrate", "sync-models", "api", "scorer", timeout=1800)
    stack.run("up", "-d", "migrate", "sync-models", timeout=600)
    deadline = time.monotonic() + 300
    while "sync-models" in stack.run("ps", "--services", "--status", "running", check=False):
        if time.monotonic() > deadline:
            raise RuntimeError("sync-models never finished")
        time.sleep(3)
    values = env_values(env_file)
    owner = (
        f"postgresql://{values['POSTGRES_USER']}:{values['POSTGRES_PASSWORD']}"
        f"@127.0.0.1:{values['POSTGRES_PORT']}/{values['POSTGRES_DB']}"
    )
    log("loading the demo's dataset")
    started = time.monotonic()
    subprocess.run(  # noqa: S603 - fixed command
        [sys.executable, "scripts/load_demo.py", SNAPSHOT],
        cwd=ROOT,
        env=os.environ | {"DATABASE_MIGRATIONS_URL": owner},
        check=True,
    )
    load_seconds = round(time.monotonic() - started, 1)
    log("starting the API at Render's limits")
    ready = start_api(stack)
    STATE.write_text(
        json.dumps(
            {
                "env_file": str(env_file),
                "compose_env": compose_env,
                "metrics_token": values["METRICS_BEARER_TOKEN"],
                "demo_load_seconds": load_seconds,
                "api_ready_seconds": ready,
            },
            indent=2,
        ),
        "utf-8",
    )
    log(f"ready: demo loaded in {load_seconds} s, API ready in {ready} s")


class Sampler:
    """docker stats every few seconds while a test runs: the container's peak memory and CPU."""

    def __init__(self, container: str, every: float = 5.0) -> None:
        self.container = container
        self.every = every
        self.samples: list[tuple[float, float]] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        while not self._stop.is_set():
            done = subprocess.run(  # noqa: S603 - fixed command
                ["docker", "stats", "--no-stream", "--format", "{{.CPUPerc}} {{.MemUsage}}",  # noqa: S607
                 self.container],
                capture_output=True, text=True, check=False,
            )  # fmt: skip
            match = re.match(r"([\d.]+)%\s+([\d.]+)(MiB|GiB)", done.stdout.strip())
            if match:
                memory = float(match.group(2)) * (1024 if match.group(3) == "GiB" else 1)
                self.samples.append((float(match.group(1)), memory))
            self._stop.wait(self.every)

    def __enter__(self) -> "Sampler":
        self._thread.start()
        return self

    def __exit__(self, *_: object) -> None:
        self._stop.set()
        self._thread.join(timeout=30)

    def summary(self) -> dict[str, Any]:
        if not self.samples:
            return {}
        cpu = [s[0] for s in self.samples]
        memory = [s[1] for s in self.samples]
        return {
            "samples": len(self.samples),
            "cpu_percent_avg": round(sum(cpu) / len(cpu), 1),
            "cpu_percent_max": round(max(cpu), 1),
            "memory_mib_first": round(memory[0], 1),
            "memory_mib_last": round(memory[-1], 1),
            "memory_mib_peak": round(max(memory), 1),
        }


def api_metrics(token: str) -> dict[str, Any]:
    """The API's own counters after a run: requests by route and status, and refusals."""
    text = httpx.get(
        f"{API}/metrics", headers=edge() | {"Authorization": f"Bearer {token}"}, timeout=30
    ).text
    wanted = ("api_predict_busy", "api_cache_events", "api_requests_total", "api_llm_calls")
    lines = [line for line in text.splitlines() if line.startswith(wanted)]
    return {"lines": lines[:200]}


def k6_numbers(summary: dict[str, Any]) -> dict[str, Any]:
    """Every sub-metric the scripts named, with the trend statistics or the count."""
    numbers: dict[str, Any] = {}
    for name, metric in summary["metrics"].items():
        values = metric.get("values", {})
        if metric.get("type") == "trend":
            numbers[name] = {
                key: round(value, 1)
                for key, value in values.items()
                if key in ("med", "p(90)", "p(95)", "p(99)", "max", "avg")
            }
        elif metric.get("type") == "counter":
            numbers[name] = int(values.get("count", 0))
        elif metric.get("type") == "rate":
            numbers[name] = round(values.get("rate", 0.0), 4)
    return numbers


def run_k6(script: str, scenario: str, label: str, extra_env: dict[str, str]) -> dict[str, Any]:
    state = load_state()
    out = WORK / f"k6-{scenario}-{label}.json"
    password = (WORK / "user_password.txt").read_text("utf-8").strip()
    env = (
        os.environ
        | {
            "API": API,
            "EDGE_SECRET": state["compose_env"]["LT_EDGE_SECRET"],
            "OWNER_PASSWORD": password,
            "OUT": str(out),
            "K6_NO_USAGE_REPORT": "true",
        }
        | extra_env
    )
    log(f"k6: {script} ({scenario}, {label})")
    started = time.monotonic()
    with Sampler(API_CONTAINER) as sampler:
        done = subprocess.run(  # noqa: S603 - fixed command
            [k6_binary(), "run", "--quiet", f"loadtest/{script}"],
            cwd=ROOT, env=env, capture_output=True, text=True, encoding="utf-8", check=False,
        )  # fmt: skip
    if not out.exists():
        print(done.stdout[-3000:], done.stderr[-3000:], file=sys.stderr)  # noqa: T201
        raise RuntimeError("k6 wrote no summary")
    summary = json.loads(out.read_text("utf-8"))
    return {
        "scenario": scenario,
        "label": label,
        "when": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "seconds": round(time.monotonic() - started, 1),
        "k6": k6_numbers(summary),
        "api_container": sampler.summary(),
        "api_metrics": api_metrics(state["metrics_token"]),
    }


def save(result: dict[str, Any]) -> None:
    path = WORK / f"{result['scenario']}-{result['label']}.json"
    path.write_text(json.dumps(result, indent=2) + "\n", "utf-8")
    log(f"saved {path.relative_to(ROOT)}")


def predict(args: argparse.Namespace) -> None:
    stack = Stack(Path(load_state()["env_file"]))
    if args.restart:
        start_api(stack, **dict(args.set or []))
    extra = {"SAMPLE_PATH": str(SAMPLE), "STAGE_SECONDS": str(args.stage_seconds)}
    save(run_k6("predict.js", "predict", args.label, extra) | {"settings": dict(args.set or [])})


def reads(args: argparse.Namespace) -> None:
    stack = Stack(Path(load_state()["env_file"]))
    if args.restart:
        start_api(stack, **dict(args.set or []))
    duration = args.duration or ("60m" if args.mode == "soak" else "5m")
    extra = {"MODE": args.mode, "DURATION": duration, "PACING": args.pacing}
    scenario = f"reads_{args.mode}" + ("_browser" if args.pacing == "browser" else "")
    result = run_k6("reads.js", scenario, args.label, extra)
    save(result | {"settings": dict(args.set or [])})


def explain(args: argparse.Namespace) -> None:
    stack = Stack(Path(load_state()["env_file"]))
    # Explanations are stored, so a second run would mostly read them back: start each from none.
    stack.psql("DELETE FROM explanations; DELETE FROM llm_usage")
    start_api(stack, LT_EXPLAIN_ENABLED="true", **dict(args.set or []))
    result = run_k6("explain.js", "explain", args.label, {})
    result["fake_llm"] = httpx.get("http://127.0.0.1:18081/stats", timeout=10).json()
    save(result | {"settings": dict(args.set or [])})
    start_api(stack)


def replay(args: argparse.Namespace) -> None:
    state = load_state()
    stack = Stack(Path(state["env_file"]))
    out = WORK / f"replay-publisher-{args.label}.json"
    scorer: dict[str, Any] = {}
    if not args.analyse_only:
        log("starting Redpanda and the scorer at the production chart's limit")
        stack.run("up", "-d", "redpanda", "scorer", timeout=900)
        code = (
            "import urllib.request;"
            "print(urllib.request.urlopen('http://127.0.0.1:8000/health',timeout=5).status)"
        )
        deadline = time.monotonic() + 900
        while "200" not in stack.run("exec", "-T", "scorer", "python", "-c", code, check=False):
            if time.monotonic() > deadline:
                raise RuntimeError("the scorer never became healthy")
            time.sleep(5)
        with Sampler(SCORER_CONTAINER, every=10) as sampler:
            subprocess.run(  # noqa: S603 - fixed command
                [sys.executable, "scripts/replay_edits.py", "--rate", str(args.rate),
                 "--minutes", str(args.minutes), "--bootstrap", "127.0.0.1:19095",
                 "--out", str(out)],
                cwd=ROOT, check=True,
            )  # fmt: skip
            published = json.loads(out.read_text("utf-8"))["edits"]
            log(f"published {published}; waiting for the scorer to catch up")
            drain_deadline = time.monotonic() + args.drain_minutes * 60
            query = f"SELECT count(*) FROM edits WHERE rev_id >= {REPLAY_OFFSET}"  # noqa: S608
            while time.monotonic() < drain_deadline:
                if int(stack.psql(query)) >= published:
                    break
                time.sleep(10)
        scorer = sampler.summary()
    save(replay_summary(stack, args.label, json.loads(out.read_text("utf-8")), scorer))


def replay_summary(
    stack: Stack, label: str, publisher: dict[str, Any], scorer: dict[str, Any]
) -> dict[str, Any]:
    """Everything from Postgres: each replayed edit's time from publish to stored score."""
    since = f"rev_id >= {REPLAY_OFFSET}"
    lag = (
        "SELECT extract(epoch FROM s.scored_at - e.edited_at) AS lag, s.scored_at, e.edited_at "  # noqa: S608
        f"FROM scores s JOIN edits e USING (rev_id) WHERE e.{since}"
    )
    numbers = stack.psql(
        "SELECT count(*), "  # noqa: S608 - REPLAY_OFFSET is a constant, not input
        "round(percentile_cont(0.5) WITHIN GROUP (ORDER BY lag)::numeric, 1), "
        "round(percentile_cont(0.95) WITHIN GROUP (ORDER BY lag)::numeric, 1), "
        "round(max(lag)::numeric, 1), "
        "round(extract(epoch FROM max(scored_at) - min(edited_at))::numeric, 1) "
        f"FROM ({lag}) t"
    ).split("|")
    per_minute = stack.psql(
        "SELECT round(percentile_cont(0.95) WITHIN GROUP (ORDER BY lag)::numeric, 1) "  # noqa: S608
        f"FROM ({lag}) t GROUP BY date_trunc('minute', edited_at) "
        "ORDER BY date_trunc('minute', edited_at)"
    ).split()
    models = stack.psql(
        "SELECT name || '=' || n FROM (SELECT model_name || ':fallback=' || used_fallback "  # noqa: S608
        f"|| ':shed=' || load_shed AS name, count(*) AS n FROM scores WHERE {since} "
        "GROUP BY 1) t ORDER BY 1"
    ).splitlines()
    unscored = stack.psql(f"SELECT count(*) FROM edits WHERE {since} AND latest_score IS NULL")  # noqa: S608
    return {
        "scenario": "replay",
        "label": label,
        "when": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "publisher": publisher,
        "scores": int(numbers[0]),
        "edit_to_score_seconds": {
            "p50": float(numbers[1]),
            "p95": float(numbers[2]),
            "max": float(numbers[3]),
        },
        "first_publish_to_last_score_seconds": float(numbers[4]),
        "p95_by_minute_of_publishing": [float(x) for x in per_minute],
        "scored_by": models,
        "stored_unscored": int(unscored),
        "scorer_container": scorer,
    }


def down(_: argparse.Namespace) -> None:
    if STATE.exists():
        state = load_state()
        Stack(Path(state["env_file"])).clean(remove_images=True)
    for name in ("state.json", "e2e.env", "user_password.txt"):
        (WORK / name).unlink(missing_ok=True)
    shutil.rmtree(WORK / "cert", ignore_errors=True)
    log("stopped")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("up")
    commands.add_parser("down")
    for name in ("predict", "reads", "explain", "replay"):
        command = commands.add_parser(name)
        command.add_argument("--label", default="before")
        command.add_argument(
            "--set", nargs=2, action="append", metavar=("NAME", "VALUE"),
            help="an LT_ setting for the API, for a tuning run (restarts it)",
        )  # fmt: skip
        command.add_argument("--restart", action="store_true", help="restart the API first")
    commands.choices["predict"].add_argument("--stage-seconds", type=int, default=120)
    commands.choices["reads"].add_argument(
        "--mode", choices=["normal", "spike", "soak"], default="normal"
    )
    commands.choices["reads"].add_argument("--duration")
    commands.choices["reads"].add_argument(
        "--pacing", choices=["fixed", "browser"], default="fixed",
        help="fixed: a clock per flow (worst case); browser: ask, answer, wait, like usePoll",
    )  # fmt: skip
    commands.choices["replay"].add_argument("--rate", type=float, default=round(5 * 74.5 / 60, 2))
    commands.choices["replay"].add_argument("--minutes", type=float, default=10)
    commands.choices["replay"].add_argument("--drain-minutes", type=float, default=20)
    commands.choices["replay"].add_argument(
        "--analyse-only", action="store_true", help="summarise the last replay from Postgres"
    )
    args = parser.parse_args()
    if getattr(args, "set", None):
        args.restart = True
    handler = {
        "up": up,
        "down": down,
        "predict": predict,
        "reads": reads,
        "explain": explain,
        "replay": replay,
    }[args.command]
    handler(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
