"""Break each thing in the failure table (PLAN.md, section 14) on the real stack, and record what
happened (Phase 8, failure injection).

    python scripts/inject_failures.py                       # every row, then clean up
    python scripts/inject_failures.py --rows 4 5 --keep     # some rows; leave the stack running
    python scripts/inject_failures.py --frontend            # also build the app and screenshot it

The stack is the end-to-end one (scripts/compose_e2e.py): every real image with its least-privilege
login, the real models in data/runs, and recorded Wikimedia fixtures served by a mock, with
docker-compose.failures.yml on top. That overlay adds the mock's control port and one FI_ switch per
failure, each off unless this script sets it. Nothing here calls Wikimedia, an LLM, DagsHub, or
Discord: the LLM hosts and the registry resolve to dead local addresses, and it reads no .env.

How each row is broken:
  1 stream down      the mock drops the stream and answers 503; edits are added while it's down
  2 rate limits      the mock answers 429 or maxlag, with Retry-After
  3 Redpanda down    docker compose stop redpanda; edits are added while it's down
  4 champion fails   the champion's model.onnx is damaged (load), then too slow for its time limit
  5 all models fail  every model too slow for its limit (run time), then every file damaged (load)
  6 Postgres down    docker compose stop postgres
  7 MLflow down      services restarted against a registry that refuses connections
  8 LLM down/budget  Explain on, with both providers unreachable; then a daily cap of 2 calls
  9 API down         docker compose stop api, under the running frontend
 10 cluster down     stop everything the cluster runs (Redpanda, ingestor, scorer, archiver)

Rows the stack can't hold: the cluster's outside alert (no UptimeRobot or Healthchecks.io here),
Prometheus alert rules firing (no Prometheus in Compose), and the trainer waiting for MLflow, which
scripts/simulate_continuous_training.py registry_down covers. Each says so in its result.

Results go to data/failure_injection.json, screenshots to docs/images/failure_*.png. Copies of
data/runs with damaged files go to data/failure_injection/ and are deleted at the end.
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent))
import compose_e2e as e2e  # noqa: E402 - shares its Stack and settings

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / "data" / "failure_injection"
MOCK = "http://127.0.0.1:18080"
API = f"http://127.0.0.1:{e2e.API_PORT}"
FRONTEND = "http://localhost:3000"
# The frontend's own threshold for "Stale" (frontend/src/lib/health.ts, STALE_SCORING_MINUTES).
STALE_MINUTES = 10
SWITCHES = (
    "FI_EXPLAIN_ENABLED",
    "FI_EXPLAIN_CAP",
    "FI_MODEL_SOURCE",
    "FI_API_TIMEOUT_MS",
    "FI_SCORER_TIMEOUT_MS",
    "FI_SCORER_POLL_SECONDS",
    "RUNS_PATH",
)


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)  # noqa: T201


class Stack(e2e.Stack):
    def __init__(self, env_file: Path) -> None:
        super().__init__(env_file)
        self.base += ["-f", "docker-compose.failures.yml"]

    def recreate(self, *services: str, **switches: str) -> None:
        """Restart services with FI_ switches set, which Compose reads from this environment."""
        for key in SWITCHES:
            os.environ.pop(key, None)
        os.environ.update(switches)
        self.run("up", "-d", "--no-deps", "--force-recreate", *services, timeout=600)

    def start_all(self) -> None:
        """Every service back as configured, with no failure switched on."""
        for key in SWITCHES:
            os.environ.pop(key, None)
        self.run("up", "-d", "--force-recreate", "api", "scorer", timeout=600)
        self.run("start", "redpanda", "postgres", "ingestor", "archiver", timeout=300)

    def metric(self, service: str, name: str) -> float | None:
        """One value from a worker's /metrics, read inside its container (no port is published)."""
        code = (
            "import urllib.request;"
            "print(urllib.request.urlopen('http://127.0.0.1:8000/metrics',timeout=5).read().decode())"
        )
        text = self.run("exec", "-T", service, "python", "-c", code, check=False, timeout=60)
        total = None
        # A counter is exposed as NAME_total, a gauge as NAME; both have a NAME_created to skip.
        wanted = {name, f"{name}_total"}
        for line in text.splitlines():
            metric = line.split("{", 1)[0].split(" ", 1)[0]
            if metric in wanted:
                total = (total or 0.0) + float(line.rsplit(" ", 1)[-1])
        return total

    def logs(self, service: str, since: str = "10m") -> str:
        return self.run("logs", "--no-log-prefix", "--since", since, service, check=False)


class Api:
    """A browser's calls, as the owner, in one tab."""

    def __init__(self, password: str) -> None:
        self.password = password
        self.client = httpx.Client(base_url=API, timeout=30)
        self.tab = f"tab-fi-{uuid.uuid4().hex[:8]}"
        self.headers: dict[str, str] = {}

    def login(self, tab: str | None = None) -> httpx.Response:
        self.tab = tab or f"tab-fi-{uuid.uuid4().hex[:8]}"
        body = {"password": self.password, "tab_id": self.tab, "take_over": True}
        reply = self.client.post("/v1/auth/login", json=body)
        if reply.status_code == 200:
            self.headers = {"x-tab-id": self.tab, "x-csrf-token": reply.json()["csrf_token"]}
        return reply

    def get(self, path: str) -> httpx.Response:
        return self.client.get(path, headers=self.headers)

    def post(self, path: str, body: dict[str, Any] | None = None) -> httpx.Response:
        return self.client.post(path, headers=self.headers, json=body or {})

    def predict(self, long: bool = False) -> httpx.Response:
        # New text each time, or the API answers from its cache and no model runs. A long one
        # (about 3,000 characters, a large diff) fills the transformers' 512 tokens, so they take
        # their full time per edit; a short one can come in under a tight time limit.
        text = f"failure injection {uuid.uuid4().hex}"
        if long:
            text += " the article was rewritten with new sources and claims" * 55
        return self.post("/v1/predict", {"added_text": text})

    def logout(self) -> None:
        try:
            self.post("/v1/auth/logout")
        except httpx.HTTPError:
            pass
        self.client.cookies.clear()
        self.headers = {}


def mock(path: str) -> dict[str, Any]:
    reply = httpx.get(f"{MOCK}{path}", timeout=10)
    reply.raise_for_status()
    return dict(reply.json())


def passed_step(line: str | None) -> bool:
    """The helper prints "STEP n ok SECONDS" or "STEP n timeout SECONDS"."""
    return bool(line) and line.split()[2] == "ok"  # type: ignore[union-attr]


def wait_until(check: Callable[[], Any], seconds: float, every: float = 2.0) -> tuple[Any, float]:
    """The first truthy result and how long it took, or (None, seconds) on timeout."""
    started = time.monotonic()
    while time.monotonic() - started < seconds:
        try:
            value = check()
        except (httpx.HTTPError, RuntimeError):
            value = None
        if value:
            return value, round(time.monotonic() - started, 1)
        time.sleep(every)
    return None, round(seconds, 1)


class Browser:
    """scripts/capture_failure_states.mjs, run beside a failure and stepped through it."""

    def __init__(self, enabled: bool, password_file: Path) -> None:
        self.enabled = enabled
        self.password_file = password_file

    def start(self, name: str, path: str, steps: list[str], owner: bool = True) -> Any:
        if not self.enabled:
            return None
        command = ["node", "scripts/capture_failure_states.mjs", "--name", name, "--path", path]
        if owner:
            command += ["--password-file", str(self.password_file)]
        for step in steps:
            command += ["--step", step]
        process = subprocess.Popen(  # noqa: S603 - fixed command
            command,
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
        )
        self.expect(process, "READY")
        return process

    @staticmethod
    def expect(process: Any, marker: str) -> str | None:
        """Read lines until one starts with the marker. None if the helper ends first."""
        if process is None:
            return None
        for line in process.stdout:
            line = line.strip()
            if line.startswith(marker):
                return line
            if line.startswith("RESULT") or "Error" in line:
                log(f"browser: {line[:300]}")
        return None

    @staticmethod
    def finish(process: Any) -> dict[str, Any] | None:
        if process is None:
            return None
        result = None
        for line in process.stdout:
            if line.startswith("RESULT "):
                result = json.loads(line[len("RESULT ") :])
        process.wait(timeout=120)
        if result is not None:
            result["screenshots"] = [Path(item).name for item in result["screenshots"]]
        return result


class Injector:
    def __init__(self, stack: Stack, password: str, browser: Browser) -> None:
        self.stack = stack
        self.api = Api(password)
        self.browser = browser

    # --- Shared checks ---

    def psql(self, query: str) -> str:
        # Queries below put revision ids in with f-strings. They're ints this script made or read
        # from the throwaway database, never outside input, so S608 is marked on each.
        return self.stack.psql(query)

    def count_in(self, rev_ids: list[int], scored: bool = False) -> int:
        where = " AND latest_score IS NOT NULL" if scored else ""
        ids = ",".join(str(rev) for rev in rev_ids)
        query = f"SELECT count(*) FROM edits WHERE rev_id IN ({ids}){where}"  # noqa: S608
        return int(self.psql(query))

    def wait_scored(self, rev_ids: list[int], seconds: float = 240) -> tuple[bool, float]:
        done, took = wait_until(
            lambda: self.count_in(rev_ids, scored=True) == len(rev_ids), seconds
        )
        return bool(done), took

    def duplicate_scores(self, rev_ids: list[int]) -> int:
        ids = ",".join(str(rev) for rev in rev_ids)
        query = (
            "SELECT count(*) FROM (SELECT rev_id, model_name FROM scores "  # noqa: S608
            f"WHERE rev_id IN ({ids}) GROUP BY 1, 2 HAVING count(*) > 1) t"
        )
        return int(self.psql(query))

    def baseline(self) -> dict[str, Any]:
        """The stack's normal state: ready, every recorded edit scored."""
        ready, took = wait_until(
            lambda: httpx.get(f"{API}/ready", timeout=5).status_code == 200, e2e.WAIT_READY_SECONDS
        )
        if not ready:
            raise RuntimeError("the API never became ready")

        def all_scored() -> bool:
            total = int(self.psql("SELECT count(*) FROM edits"))
            done = int(self.psql("SELECT count(*) FROM edits WHERE latest_score IS NOT NULL"))
            return total >= 10 and done == total

        scored, scored_after = wait_until(all_scored, e2e.WAIT_SCORED_SECONDS)
        if not scored:
            raise RuntimeError("the recorded edits were never all scored")
        return {
            "api_ready_seconds": took,
            "edits_scored": int(self.psql("SELECT count(*) FROM edits")),
            "scored_after_seconds": scored_after,
        }

    # --- Rows ---

    def row1_stream_down(self) -> dict[str, Any]:
        reconnects_before = self.stack.metric("ingestor", "ingestor_reconnects") or 0.0
        connects_before = len(mock("/stats")["stream_connects"])
        browser = self.browser.start(
            "stream_down", "/feed", ["text:Stale", "gone:No edit has been scored"]
        )
        mock("/control?stream=down")
        down_at = time.monotonic()
        added = mock("/control?add=5")["added"]
        hold = STALE_MINUTES * 60 + 90 if browser else 150
        log(f"row 1: stream down for {hold} s")
        stale_seen = self.browser.expect(browser, "STEP 1")
        while time.monotonic() - down_at < hold:
            time.sleep(5)
        ingestor_health = self.stack.run(
            "exec",
            "-T",
            "ingestor",
            "python",
            "-c",
            "import urllib.request as u\ntry: print(u.urlopen('http://127.0.0.1:8000/health').status)"
            "\nexcept Exception as e: print(getattr(e, 'code', 'error'))",
            check=False,
        ).strip()
        stats = mock("/stats")
        attempts = [c for c in stats["stream_connects"][connects_before:] if c["answer"] == 503]
        gaps = [round(b["at"] - a["at"], 1) for a, b in zip(attempts, attempts[1:], strict=False)]
        reconnects_down = (self.stack.metric("ingestor", "ingestor_reconnects") or 0.0) - (
            reconnects_before
        )
        mock("/control?stream=up")
        up_at = time.monotonic()
        repaired, took = self.wait_scored(added)
        resumed = [c for c in mock("/stats")["stream_connects"] if c["at"] >= time.time() - 600]
        ui = None
        if browser:
            self.browser.expect(browser, "STEP 2")
            ui = self.browser.finish(browser)
        return {
            "outage_seconds": round(up_at - down_at, 1),
            "reconnect_attempts_while_down": len(attempts),
            "seconds_between_attempts": gaps,
            "reconnects_metric_increase": reconnects_down,
            "ingestor_health_while_down": ingestor_health,
            "edits_added_during_outage": len(added),
            "gap_repaired": f"{self.count_in(added)}/{len(added)} stored, "
            f"{self.count_in(added, scored=True)}/{len(added)} scored",
            "resumed_with_last_event_id": any(c.get("last_event_id") for c in resumed),
            "repair_seconds_after_stream_back": took,
            "duplicate_scores": self.duplicate_scores(added),
            "ui": ui,
            "ui_stale_banner_seen": passed_step(stale_seen),
            "passed": repaired and len(attempts) >= 2,
        }

    def _throttle(self, kind: str, count: int, wait: int, edits: int) -> dict[str, Any]:
        calls_before = len(mock("/stats")["api_calls"])
        dead_before = self.stack.metric("ingestor", "ingestor_dead_letters") or 0.0
        mock(f"/control?api={kind}&count={count}&wait={wait}")
        added = mock(f"/control?add={edits}")["added"]
        arrived, took = self.wait_scored(added, seconds=180)
        calls = mock("/stats")["api_calls"][calls_before:]
        mock("/control?api=ok")
        honored = []
        for before, after in zip(calls, calls[1:], strict=False):
            if before["answer"] == kind:
                honored.append(round(after["at"] - before["at"], 2))
        dead = (self.stack.metric("ingestor", "ingestor_dead_letters") or 0.0) - dead_before
        return {
            "injected": f"{count} x {kind}, Retry-After {wait} s",
            "edits_added": len(added),
            "edits_scored": self.count_in(added, scored=True),
            "seconds_to_score": took,
            "seconds_after_each_refusal": honored,
            "retry_after_honored": bool(honored) and min(honored) >= wait - 0.05,
            "every_call_sent_maxlag": all(call["maxlag"] for call in calls),
            "dead_lettered": dead,
            "all_arrived": bool(arrived),
        }

    def row2_rate_limits(self) -> dict[str, Any]:
        short_429 = self._throttle("429", 3, 3, 2)
        short_maxlag = self._throttle("maxlag", 3, 3, 2)
        # Longer than the client's 6 attempts: what happens to the one edit that can't get a diff.
        sustained = self._throttle("429", 8, 1, 2)
        lag = self.stack.metric("ingestor", "ingestor_lag_seconds")
        rules = (ROOT / "k8s").rglob("*.yaml")
        lag_rules = sorted(
            {
                match
                for path in rules
                for match in re.findall(r"alert:\s*(\w*Ingest\w*)", path.read_text("utf-8"))
            }
        )
        return {
            "short_429": short_429,
            "short_maxlag": short_maxlag,
            "sustained_429": sustained,
            "ingestor_lag_seconds_after": lag,
            "lag_alert_rules_in_chart": lag_rules,
            "alert_fired": "not run: Compose has no Prometheus (the rules are tested on kind)",
            "passed": short_429["retry_after_honored"]
            and short_maxlag["retry_after_honored"]
            and short_429["all_arrived"]
            and short_maxlag["all_arrived"],
        }

    def row3_redpanda_down(self) -> dict[str, Any]:
        self.stack.run("stop", "redpanda", timeout=120)
        down_at = time.monotonic()
        added = mock("/control?add=5")["added"]
        time.sleep(90)
        running = self.stack.run("ps", "--format", "{{.Service}} {{.State}}", check=False)
        states = dict(line.split(" ", 1) for line in running.strip().splitlines() if " " in line)
        stored_while_down = self.count_in(added)
        ingestor_log = self.stack.logs("ingestor", "2m")
        self.stack.run("start", "redpanda", timeout=180)
        up_at = time.monotonic()
        repaired, took = self.wait_scored(added, seconds=300)
        return {
            "outage_seconds": round(up_at - down_at, 1),
            "edits_added_during_outage": len(added),
            "stored_while_down": stored_while_down,
            "service_states_while_down": {
                key: states.get(key) for key in ("ingestor", "scorer", "archiver", "api")
            },
            "ingestor_logged_interruption": "Stream interrupted" in ingestor_log,
            "gap_repaired": f"{self.count_in(added, scored=True)}/{len(added)} scored",
            "repair_seconds_after_redpanda_back": took,
            "duplicate_scores": self.duplicate_scores(added),
            "buffer_limit_messages": 10_000,
            # Every service rides it out: the archiver used to exit here (F-161).
            "passed": repaired
            and stored_while_down == 0
            and all(states.get(key) == "running" for key in ("ingestor", "scorer", "archiver")),
        }

    def _broken_runs(self, name: str, models: list[str]) -> Path:
        """A copy of data/runs with model.onnx damaged and checksums.json left as it was."""
        target = WORK / name
        shutil.rmtree(target, ignore_errors=True)
        target.mkdir(parents=True)
        shutil.copy2(ROOT / "data/runs/aliases.json", target / "aliases.json")
        for model in ("roberta", "distilbert", "bilstm", "bert", "mlp"):
            shutil.copytree(ROOT / "data/runs" / model / "model", target / model / "model")
        for model in models:
            onnx = target / model / "model" / "model.onnx"
            data = bytearray(onnx.read_bytes())
            data[:4096] = os.urandom(4096)
            onnx.write_bytes(bytes(data))
        return target

    def _unthrottle(self) -> None:
        for service in ("api", "scorer"):
            subprocess.run(  # noqa: S603 - fixed command
                ["docker", "update", "--cpus", "0", f"{e2e.PROJECT}-{service}-1"],  # noqa: S607
                check=False,
                capture_output=True,
            )

    def _clear_model_caches(self) -> None:
        for service in ("api", "scorer"):
            self.stack.run(
                "run", "--rm", "--no-deps", "--entrypoint", "sh", service, "-c",
                "rm -rf /models/cache/*", check=False, timeout=180,
            )  # fmt: skip

    def _ready(self, seconds: float = 300) -> tuple[bool, float]:
        done, took = wait_until(
            lambda: httpx.get(f"{API}/ready", timeout=5).status_code == 200, seconds
        )
        return bool(done), took

    def _latest_scores(self, rev_ids: list[int]) -> dict[str, int]:
        ids = ",".join(str(rev) for rev in rev_ids)
        rows = self.psql(
            "SELECT model_name || ':' || used_fallback || '=' || count(*) FROM scores "  # noqa: S608
            f"WHERE rev_id IN ({ids}) GROUP BY model_name, used_fallback"
        )
        return dict(line.split("=") for line in rows.splitlines() if "=" in line)

    def _restore_models(self) -> dict[str, Any]:
        self.stack.recreate("api", "scorer")
        ready, took = self._ready()
        return {"ready_again": ready, "seconds": took}

    def row4_champion_fails(self) -> dict[str, Any]:
        result: dict[str, Any] = {}
        # (a) The champion's file is damaged: found when it's checked at startup.
        runs = self._broken_runs("runs-champion-broken", ["roberta"])
        self._clear_model_caches()
        self.stack.recreate("api", "scorer", RUNS_PATH=runs.as_posix())
        ready, took = self._ready()
        self.api.login()
        answer = self.api.predict()
        health = self.api.get("/v1/health/summary").json()
        added = mock("/control?add=3")["added"]
        scored, _ = self.wait_scored(added)
        result["damaged_file"] = {
            "api_ready": ready,
            "ready_seconds": took,
            "predict": {k: answer.json().get(k) for k in ("model_name", "used_fallback")},
            "health_chain": [(c["alias"], c["model_name"]) for c in health["chain"]],
            "scorer_scores": self._latest_scores(added),
            "edits_scored": scored,
            "api_log_mentions_champion": "Left champion out" in self.stack.logs("api", "5m"),
        }
        self.api.logout()
        # (b) Healthy files, but the champion is slower than its time limit on every edit. The
        # limit is set below RoBERTa's time per edit and above the others' (docs/metrics.md).
        self._clear_model_caches()
        self.stack.recreate("api", "scorer", FI_SCORER_TIMEOUT_MS="40", FI_API_TIMEOUT_MS="40")
        self._ready()
        self.api.login()
        answers = [self.api.predict(long=True).json() for _ in range(8)]
        health = self.api.get("/v1/health/summary").json()
        added = mock("/control?add=6")["added"]
        scored, _ = self.wait_scored(added)
        result["too_slow"] = {
            "predict_answers": sorted(
                {f"{a.get('model_name')}:fallback={a.get('used_fallback')}" for a in answers}
            ),
            "breakers": {c["model_name"]: c["breaker"] for c in health["chain"]},
            "scorer_scores": self._latest_scores(added),
            "edits_scored": scored,
        }
        self.api.logout()
        result["restored"] = self._restore_models()
        load = result["damaged_file"]
        slow = result["too_slow"]
        result["passed"] = (
            load["api_ready"]
            and load["predict"]["used_fallback"] is True
            and load["edits_scored"]
            and all("fallback=True" in item for item in slow["predict_answers"])
        )
        return result

    def row5_all_models_fail(self) -> dict[str, Any]:
        result: dict[str, Any] = {}
        # (a) At run time. A 1 ms limit alone isn't enough: the scorer's limit is per edit, summed
        # over a micro-batch, and the MLP can meet it. So the scorer is also starved of CPU, as on
        # an overloaded node. CPU limits work in 100 ms slices, so a model can still land inside
        # one: this checks what must hold either way, not that every call fails. 0.05 CPU, not
        # less: at 0.01 Docker Desktop couldn't stop the container, and requests timed out.
        # The API isn't starved: its run-time answers are recorded, and part (b) shows its 503.
        self.stack.recreate("api", "scorer", FI_SCORER_TIMEOUT_MS="1", FI_API_TIMEOUT_MS="1")
        self._ready()
        dlq_before = self.stack.metric("scorer", "scorer_dlq") or 0.0
        subprocess.run(  # noqa: S603 - fixed command
            ["docker", "update", "--cpus", "0.05", f"{e2e.PROJECT}-scorer-1"],  # noqa: S607
            check=True,
            capture_output=True,
        )
        self.api.login()
        replies = [self.api.predict(long=True) for _ in range(3)]
        added = mock("/control?add=4")["added"]
        wait_until(lambda: self.count_in(added) == len(added), 300)
        time.sleep(25)  # two rescore rounds (SCORER_RESCORE_SECONDS=10) that should still fail
        result["run_time"] = {
            "injected": "PREDICT_TIMEOUT_MS=1 in both, and the scorer held to 0.05 CPU",
            "predict_statuses": [reply.status_code for reply in replies],
            "predict_503_body": next(
                (
                    {k: r.json().get(k) for k in ("code", "title")}
                    for r in replies
                    if r.status_code == 503
                ),
                None,
            ),
            "edits_stored": self.count_in(added),
            "edits_scored": self.count_in(added, scored=True),
            "stored_unscored_metric": self.stack.metric("scorer", "scorer_stored_unscored"),
            "dead_lettered": (self.stack.metric("scorer", "scorer_dlq") or 0.0) - dlq_before,
        }
        self.api.logout()
        # Mend it: the stored edits should be rescored from Postgres, not replayed from Redpanda.
        # CPU back first: a container held to 0.01 CPU can't shut down in time, and Docker's kill
        # then leaves a zombie it can't remove.
        self._unthrottle()
        self.stack.recreate("api", "scorer")
        rescored, took = self.wait_scored(added, seconds=180)
        result["rescored_after_fix"] = {
            "all": rescored,
            "seconds": took,
            "rescored_metric": self.stack.metric("scorer", "scorer_rescored"),
        }
        # (b) At startup: every model file damaged, so nothing can load.
        runs = self._broken_runs(
            "runs-all-broken", ["roberta", "distilbert", "bilstm", "bert", "mlp"]
        )
        self._clear_model_caches()
        self.stack.recreate("api", "scorer", RUNS_PATH=runs.as_posix())
        waiting = mock("/control?add=3")["added"]
        time.sleep(60)
        states = self.stack.run("ps", "-a", "--format", "{{.Service}} {{.State}} {{.Status}}")
        ready = httpx.get(f"{API}/ready", timeout=5)
        self.api.login()
        reply = self.api.predict()
        queue = self.api.get("/v1/queue")
        result["load_time"] = {
            "service_states": [
                line for line in states.splitlines() if line.split(" ")[0] in {"api", "scorer"}
            ],
            "ready_status": ready.status_code,
            "predict_status": reply.status_code,
            "queue_status": queue.status_code,
            "edits_stored_while_no_model": self.count_in(waiting),
            "scorer_log_tail": self.stack.logs("scorer", "2m").strip().splitlines()[-1:],
        }
        self.api.logout()
        result["restored"] = self._restore_models()
        caught_up, took = self.wait_scored(waiting, seconds=180)
        result["load_time"] |= {"scored_once_models_back": caught_up, "seconds": took}
        run_time, load_time = result["run_time"], result["load_time"]
        # Whether a model still answered inside its slice of CPU is luck, so reaching the unscored
        # path is recorded, not required. It's tested directly in services/scorer/tests (F-157).
        run_time["unscored_path_reached"] = (run_time["stored_unscored_metric"] or 0) >= 1
        result["passed"] = (
            run_time["edits_stored"] == len(added)
            and run_time["dead_lettered"] == 0
            and rescored
            and load_time["predict_status"] == 503
            and load_time["queue_status"] == 200
            and caught_up
        )
        return result

    def row6_postgres_down(self) -> dict[str, Any]:
        result: dict[str, Any] = {}
        first = self.api.login()
        self.api.get("/v1/queue")
        self.api.get("/v1/health/summary")
        self.stack.run("stop", "postgres", timeout=120)
        down_at = time.monotonic()
        queue = self.api.get("/v1/queue")
        health = self.api.get("/v1/health/summary")
        other = Api(self.api.password)
        new_login = other.login()
        # The current session, through its lease (GATE_LEASE_SECONDS, 45 s by default).
        during = []
        for _ in range(7):
            reply = self.api.get("/v1/queue")
            stale = reply.json().get("stale") if reply.status_code == 200 else None
            during.append((round(time.monotonic() - down_at), reply.status_code, stale))
            time.sleep(10)
        added = mock("/control?add=4")["added"]
        time.sleep(20)
        scorer_lag = self.stack.metric("scorer", "scorer_consumer_lag")
        retries = self.stack.metric("scorer", "scorer_delivery_retries")
        result["api"] = {
            "signed_in_before": first.status_code,
            "queue_status": queue.status_code,
            "queue_stale_flag": queue.json().get("stale") if queue.status_code == 200 else None,
            "health_stale_flag": health.json().get("stale") if health.status_code == 200 else None,
            "new_sign_in_status": new_login.status_code,
            "new_sign_in_code": new_login.json().get("code"),
            "current_session_over_time": during,
            "ready_status": httpx.get(f"{API}/ready", timeout=10).status_code,
        }
        result["scorer"] = {"consumer_lag_while_down": scorer_lag, "write_retries": retries}
        self.stack.run("start", "postgres", timeout=180)
        up_at = time.monotonic()
        recovered, took = self.wait_scored(added, seconds=300)
        result["scorer"] |= {
            "outage_seconds": round(up_at - down_at, 1),
            "edits_added_during_outage": len(added),
            "scored_after_postgres_back": recovered,
            "seconds_to_catch_up": took,
            "duplicate_scores": self.duplicate_scores(added),
        }
        relogin, relogin_took = wait_until(lambda: self.api.login().status_code == 200, 120)
        result["api"]["sign_in_works_again_after_seconds"] = relogin_took if relogin else None
        self.api.logout()
        # The same outage as a signed-in owner sees it.
        browser = self.browser.start("postgres_down", "/queue", ["text:."])
        if browser:
            self.browser.expect(browser, "STEP 1")
            self.browser.finish(browser)
            browser = self.browser.start(
                "postgres_down",
                "/queue",
                ["text:Showing cached data|database", "gone:Showing cached"],
            )
            self.stack.run("stop", "postgres", timeout=120)
            seen = self.browser.expect(browser, "STEP 1")
            self.stack.run("start", "postgres", timeout=180)
            self.browser.expect(browser, "STEP 2")
            result["ui"] = self.browser.finish(browser) | {"degraded_state_seen": passed_step(seen)}
            self.api.login()
            self.api.logout()
        api = result["api"]
        # Served stale from cache while the lease lasts, refused once it's over.
        served_stale = any(code == 200 and stale for _, code, stale in during)
        refused_after_lease = during[-1][1] >= 500
        result["passed"] = (
            api["queue_status"] == 200
            and served_stale
            and refused_after_lease
            and api["new_sign_in_status"] >= 500
            and recovered
            and result["scorer"]["duplicate_scores"] == 0
        )
        return result

    def row7_mlflow_down(self) -> dict[str, Any]:
        # Warm caches first: a normal start saves the chain beside the models.
        self.stack.recreate("api", "scorer")
        self._ready()
        self.stack.recreate("api", "scorer", FI_MODEL_SOURCE="mlflow", FI_SCORER_POLL_SECONDS="5")
        ready, took = self._ready()
        self.api.login()
        reply = self.api.predict()
        added = mock("/control?add=3")["added"]
        scored, _ = self.wait_scored(added)
        time.sleep(20)  # at least three alias polls against the dead registry
        failures = self.stack.metric("scorer", "scorer_reload_failures")
        api_log = self.stack.logs("api", "5m")
        self.api.logout()
        result = {
            "api_ready": ready,
            "ready_seconds": took,
            "api_used_saved_chain": "using the saved chain" in api_log,
            "predict": {k: reply.json().get(k) for k in ("model_name", "used_fallback")},
            "scorer_scored_new_edits": scored,
            "scorer_reload_failures": failures,
            "training_waits": "see scripts/simulate_continuous_training.py registry_down",
        }
        result["restored"] = self._restore_models()
        result["passed"] = ready and scored and bool(failures) and result["api_used_saved_chain"]
        return result

    def row8_llm_down(self) -> dict[str, Any]:
        rev_ids = [int(r) for r in self.psql(
            "SELECT rev_id FROM edits WHERE added_text IS NOT NULL ORDER BY rev_id LIMIT 3"
        ).splitlines()]  # fmt: skip
        self.stack.recreate("api", FI_EXPLAIN_ENABLED="true", FI_EXPLAIN_CAP="50")
        self._ready()
        self.api.login()
        down = self.api.post(f"/v1/explain/{rev_ids[0]}")
        others = {
            "queue": self.api.get("/v1/queue").status_code,
            "predict": self.api.predict().status_code,
            "edit": self.api.get(f"/v1/edits/{rev_ids[0]}").status_code,
        }
        self.api.logout()
        result: dict[str, Any] = {
            "providers_down": {
                "status": down.status_code,
                "body": {k: down.json().get(k) for k in ("explanation", "available", "reason")},
                "other_endpoints": others,
            }
        }
        if self.browser.enabled:
            browser = self.browser.start(
                "llm_down",
                f"/edits/{rev_ids[1]}",
                ["click:Explain", "text:didn.t answer|Explanation unavailable"],
            )
            self.browser.expect(browser, "STEP 2")
            result["ui"] = self.browser.finish(browser)
        # Over budget: the cap counts every reserved call, failed ones too.
        self.psql("DELETE FROM llm_usage")
        self.stack.recreate("api", FI_EXPLAIN_ENABLED="true", FI_EXPLAIN_CAP="2")
        self._ready()
        self.api.login()
        capped = [self.api.post(f"/v1/explain/{rev}").json() for rev in rev_ids]
        usage = self.psql("SELECT provider || '=' || calls FROM llm_usage ORDER BY provider")
        predict_ok = self.api.predict().status_code
        self.api.logout()
        result["over_budget"] = {
            "answers": [(a.get("available"), a.get("reason")) for a in capped],
            "llm_usage_rows": usage.splitlines(),
            "predict_status": predict_ok,
        }
        self.stack.recreate("api")
        self._ready()
        body = result["providers_down"]["body"]
        result["passed"] = (
            down.status_code == 200
            and body["available"] is False
            and body["explanation"] == "Explanation unavailable"
            and set(others.values()) == {200}
            and any(reason == "daily_cap" for _, reason in result["over_budget"]["answers"])
        )
        return result

    def row9_api_down(self) -> dict[str, Any]:
        result: dict[str, Any] = {}
        self.stack.run("stop", "api", timeout=120)
        down_at = time.monotonic()
        browser = self.browser.start(
            "api_down", "/queue", ["text:isn.t answering", "text:Continue as guest"], owner=False
        )
        seen = self.browser.expect(browser, "STEP 1")
        result["waking_state_seen_after_seconds"] = (
            float(seen.split()[-1]) if passed_step(seen) else None
        )
        time.sleep(20)
        self.stack.run("start", "api", timeout=180)
        up_at = time.monotonic()
        ready, took = self._ready()
        recovered = self.browser.expect(browser, "STEP 2")
        result |= {
            "outage_seconds": round(up_at - down_at, 1),
            "api_ready_seconds_after_start": took,
            "ui_recovered_without_reload": passed_step(recovered),
            "ui": self.browser.finish(browser),
        }
        if not self.browser.enabled:
            result["ui"] = "not run: needs --frontend"
        result["passed"] = ready and (
            not self.browser.enabled or passed_step(seen) and passed_step(recovered)
        )
        return result

    def row10_cluster_down(self) -> dict[str, Any]:
        cluster = ("ingestor", "scorer", "archiver", "redpanda")
        self.stack.run("stop", *cluster, timeout=180)
        down_at = time.monotonic()
        added = mock("/control?add=3")["added"]
        self.api.login()
        latest = self.psql("SELECT max(rev_id) FROM edits WHERE latest_score IS NOT NULL")
        reads = {
            "queue": self.api.get("/v1/queue").status_code,
            "feed": self.api.get("/v1/feed").status_code,
            "edit": self.api.get(f"/v1/edits/{latest}").status_code,
            "models": self.api.get("/v1/models").status_code,
            "predict": self.api.predict().status_code,
        }
        items = len(self.api.get("/v1/queue").json().get("items", []))
        self.api.logout()
        # Only now: one seat, so the reads above, signed in with take-over, would end its session.
        browser = self.browser.start("cluster_down", "/feed", ["text:Stale", "text:Delayed"])
        stale_seen = None
        if browser:
            stale_seen = self.browser.expect(browser, "STEP 1")
        else:
            time.sleep(60)
        delayed_seen = self.browser.expect(browser, "STEP 2") if browser else None
        outage = round(time.monotonic() - down_at, 1)
        self.stack.run("start", "redpanda", timeout=180)
        self.stack.run("start", "ingestor", "scorer", "archiver", timeout=180)
        repaired, took = self.wait_scored(added, seconds=300)
        return {
            "stopped": list(cluster),
            "outage_seconds": outage,
            "api_reads_while_down": reads,
            "queue_items_while_down": items,
            "ui_stale_banner_seen": passed_step(stale_seen),
            "ui_feed_pill_delayed_seen": passed_step(delayed_seen),
            "ui": self.browser.finish(browser),
            "outside_alert": "not run: UptimeRobot and Healthchecks.io aren't set up (DEC-091)",
            "edits_added_during_outage": len(added),
            "gap_repaired_after_restart": repaired,
            "repair_seconds": took,
            "passed": set(reads.values()) == {200} and items > 0 and repaired,
        }


ROWS: dict[int, tuple[str, str]] = {
    1: ("Wikimedia stream down", "row1_stream_down"),
    2: ("Wikimedia API rate limits", "row2_rate_limits"),
    3: ("Redpanda down", "row3_redpanda_down"),
    4: ("Champion fails", "row4_champion_fails"),
    5: ("All models fail", "row5_all_models_fail"),
    6: ("Postgres down", "row6_postgres_down"),
    7: ("MLflow/DagsHub down", "row7_mlflow_down"),
    8: ("LLM down or over budget", "row8_llm_down"),
    9: ("API asleep or down", "row9_api_down"),
    10: ("Cluster down", "row10_cluster_down"),
}


def start_frontend(log_path: Path) -> subprocess.Popen[bytes]:
    """Build the app against the stack's API and serve it on :3000, like the screenshot script."""
    npm = "npm.cmd" if os.name == "nt" else "npm"
    env = os.environ | {"API_ORIGIN": API}
    log("building the frontend")
    subprocess.run(  # noqa: S603 - fixed command
        [npm, "run", "build"], cwd=ROOT / "frontend", env=env, check=True, capture_output=True
    )
    output = log_path.open("wb")
    process = subprocess.Popen(  # noqa: S603 - fixed command
        [npm, "start"], cwd=ROOT / "frontend", env=env, stdout=output, stderr=output
    )
    ready, _ = wait_until(lambda: httpx.get(FRONTEND, timeout=5).status_code == 200, 120)
    if not ready:
        raise RuntimeError("the frontend didn't start")
    return process


def stop_process_tree(process: subprocess.Popen[bytes] | None) -> None:
    if process is None or process.poll() is not None:
        return
    if os.name == "nt":
        # npm starts node as a child; terminating npm alone leaves node holding the port (F-089).
        subprocess.run(  # noqa: S603 - fixed command
            ["taskkill", "/T", "/F", "/PID", str(process.pid)],  # noqa: S607
            capture_output=True,
            check=False,
        )
    else:
        process.terminate()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--rows", type=int, nargs="*", default=sorted(ROWS))
    parser.add_argument("--keep", action="store_true", help="leave the stack running")
    parser.add_argument("--frontend", action="store_true", help="build the app and screenshot it")
    parser.add_argument("--out", type=Path, default=ROOT / "data" / "failure_injection.json")
    args = parser.parse_args()

    WORK.mkdir(parents=True, exist_ok=True)
    folder = Path(tempfile.mkdtemp(prefix="wikiguard-fi-"))
    env_file, password = e2e.write_env(folder)
    password_file = WORK / "user_password.txt"
    e2e.write_private(password_file, password)
    stack = Stack(env_file)
    frontend = None
    results: dict[str, Any] = {"started": time.strftime("%Y-%m-%dT%H:%M:%S"), "rows": {}}
    previous = json.loads(args.out.read_text("utf-8")) if args.out.exists() else {}
    if set(args.rows) != set(ROWS):
        results["rows"] = previous.get("rows", {})
    try:
        stack.clean()
        log("building and starting the stack")
        stack.run("up", "-d", "--build", timeout=1800)
        injector = Injector(stack, password, Browser(args.frontend, password_file))
        results["baseline"] = injector.baseline()
        log(f"baseline: {results['baseline']}")
        if args.frontend:
            frontend = start_frontend(WORK / "frontend.log")
        for row in args.rows:
            title, method = ROWS[row]
            log(f"row {row}: {title}")
            started = time.monotonic()
            try:
                outcome = getattr(injector, method)()
            except Exception as error:  # noqa: BLE001 - record it and carry on with the next row
                outcome = {"passed": False, "error": f"{type(error).__name__}: {error}"[:500]}
                # Best effort, so one row's failure can't stop the rows after it.
                try:
                    injector._unthrottle()
                    stack.start_all()
                except RuntimeError as again:
                    log(f"couldn't restore the stack after row {row}: {again}")
            outcome["seconds"] = round(time.monotonic() - started, 1)
            results["rows"][str(row)] = {"failure": title} | outcome
            log(f"row {row}: {'passed' if outcome.get('passed') else 'FAILED'}")
            args.out.write_text(json.dumps(results, indent=2, default=str) + "\n", "utf-8")
    finally:
        stop_process_tree(frontend)
        for key in SWITCHES:
            os.environ.pop(key, None)
        if not args.keep:
            stack.clean(remove_images=True)
            shutil.rmtree(WORK, ignore_errors=True)
        env_file.unlink(missing_ok=True)
    args.out.write_text(json.dumps(results, indent=2, default=str) + "\n", "utf-8")
    print(json.dumps(results, indent=2, default=str))  # noqa: T201
    failed = [key for key, row in results["rows"].items() if not row.get("passed")]
    if failed:
        print(f"Rows that didn't behave as the table says: {failed}", file=sys.stderr)  # noqa: T201
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
