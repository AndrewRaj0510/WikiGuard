"""Measure what one visitor costs each free tier, for docs/cost.md (PLAN.md, Phase 8).

    python scripts/load_test.py up
    python scripts/measure_usage.py [--seconds 120] [--skip-build]
    python scripts/load_test.py down

The demo as it will run: the API at Render's limits on the demo's data (the load test's stack),
the frontend built as Vercel builds it, and the real Worker in front (scripts/edge_local.mjs). A
browser (scripts/measure_usage.mjs) sits on the landing page, then on each console page as a
guest, and counts its requests. Meanwhile this samples the network counters inside the API's and
Postgres's containers: what the API sends stands in for Render's outbound bandwidth, and what
Postgres sends to the API for Supabase's egress. The local Postgres isn't Supabase and the API
here doesn't sleep, so these are per-minute rates, not a month's bill.

Writes data/usage.json.
"""

import argparse
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import check_logs  # noqa: E402 - its frontend and Worker helpers
import compose_e2e as e2e  # noqa: E402
import load_test as lt  # noqa: E402
import pentest  # noqa: E402

ROOT = lt.ROOT
OUT = ROOT / "data" / "usage.json"
CONTAINERS = {"api": lt.API_CONTAINER, "postgres": f"{e2e.PROJECT}-postgres-1"}


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)  # noqa: T201


def sent_bytes(container: str) -> int:
    done = subprocess.run(  # noqa: S603 - fixed command
        ["docker", "exec", container, "cat", "/sys/class/net/eth0/statistics/tx_bytes"],  # noqa: S607
        capture_output=True,
        text=True,
        check=True,
    )
    return int(done.stdout.strip())


class Counters(threading.Thread):
    """Each container's bytes sent, every 2 s, with the wall-clock time in milliseconds."""

    def __init__(self) -> None:
        super().__init__(daemon=True)
        self.samples: list[tuple[float, dict[str, int]]] = []
        self.done = threading.Event()

    def run(self) -> None:
        while not self.done.is_set():
            self.samples.append(
                (time.time() * 1000, {name: sent_bytes(c) for name, c in CONTAINERS.items()})
            )
            self.done.wait(2)

    def at(self, moment_ms: float) -> dict[str, int]:
        """The last sample at or before a moment."""
        before = [values for at, values in self.samples if at <= moment_ms]
        return before[-1] if before else self.samples[0][1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seconds", type=int, default=120, help="time on each page")
    parser.add_argument("--skip-build", action="store_true", help="use the frontend's last build")
    args = parser.parse_args()
    state = lt.load_state()
    stack = lt.Stack(Path(state["env_file"]))
    npm = "npm.cmd" if os.name == "nt" else "npm"
    env = {key: value for key, value in os.environ.items() if key != "API_ORIGIN"}
    if not args.skip_build:
        log("building the frontend as Vercel does")
        subprocess.run(  # noqa: S603 - fixed command
            [npm, "run", "build"], cwd=ROOT / "frontend", env=env, check=True, capture_output=True
        )
    lt.start_api(stack, LT_CORS_ORIGINS=pentest.EDGE)
    stack.psql("DELETE FROM active_session; DELETE FROM login_throttle")
    edit = stack.psql("SELECT rev_id FROM edits ORDER BY latest_score DESC LIMIT 1")
    server = subprocess.Popen(  # noqa: S603 - fixed command
        [npm, "start"],
        cwd=ROOT / "frontend",
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    worker = pentest.start_edge(state["compose_env"]["LT_EDGE_SECRET"])
    counters = Counters()
    try:
        e2e.wait_for("the frontend", check_logs.frontend_up, 120)
        counters.start()
        log(f"a visitor on 6 pages, {args.seconds} s each")
        done = subprocess.run(  # noqa: S603 - fixed command
            ["node", "scripts/measure_usage.mjs", "--seconds", str(args.seconds), "--edit", edit],  # noqa: S607
            cwd=ROOT,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
        line = next((x for x in done.stdout.splitlines() if x.startswith("RESULT ")), None)
        if line is None:
            raise RuntimeError(f"the browser failed: {done.stderr[-1500:]}")
        browser = json.loads(line.removeprefix("RESULT "))
    finally:
        counters.done.set()
        if os.name == "nt":
            subprocess.run(  # noqa: S603 - fixed command; before terminate, or the server lives on
                ["taskkill", "/F", "/T", "/PID", str(server.pid)],  # noqa: S607
                capture_output=True,
                check=False,
            )
        server.terminate()
        worker.terminate()
    load_ms = browser["load_seconds"] * 1000
    minutes = (browser["seconds"] - browser["load_seconds"]) / 60
    for phase in browser["phases"]:
        start, steady, end = (
            phase["started_ms"],
            phase["started_ms"] + load_ms,
            phase["ended_ms"],
        )
        a, b, c = counters.at(start), counters.at(steady), counters.at(end)
        phase["containers_load_bytes"] = {name: b[name] - a[name] for name in CONTAINERS}
        phase["containers_bytes_per_minute"] = {
            name: round((c[name] - b[name]) / minutes) for name in CONTAINERS
        }
    result = {"when": time.strftime("%Y-%m-%dT%H:%M:%S"), **browser}
    OUT.write_text(json.dumps(result, indent=2) + "\n", "utf-8")
    for phase in browser["phases"]:
        log(
            f"{phase['name']}: load {phase['load']['worker']} requests; per minute "
            f"{phase['steady_per_minute']['worker']} requests, API sent "
            f"{phase['containers_bytes_per_minute']['api']} B, Postgres sent "
            f"{phase['containers_bytes_per_minute']['postgres']} B"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
