"""Rehearse a scheduled rotation of every secret the API holds, then a security incident, on the
load test's stack behind the real Worker (PLAN.md, Phase 8; docs/runbooks/secret-rotation.md and
docs/runbooks/security-incident.md).

    python scripts/load_test.py up
    python scripts/rehearse_incident.py
    python scripts/load_test.py down

The stack's own env file (data/load_test/e2e.env) stands in for Render's environment page, and
restarting the local Worker stands in for `wrangler secret put`. A rotation changes a value there
and recreates the API, as a Render deploy does. A visitor asks the API through the Worker four
times a second throughout, so every refused or failed request is counted.

Locally a recreate stops the old container before the new one is ready, so the gaps marked
"restart" are the laptop's: Render keeps the old instance serving until the new one passes its
health check. A 403 is what the procedure itself costs visitors, and that's the number to judge.

Every value here is random per run and opens only this throwaway stack. Writes
data/rehearsal.json. Calls nothing outside the laptop.
"""

import argparse
import json
import os
import secrets
import subprocess
import sys
import threading
import time
from collections import Counter
from pathlib import Path
from typing import Any

import httpx
import psycopg
from argon2 import PasswordHasher
from psycopg import sql

sys.path.insert(0, str(Path(__file__).resolve().parent))
import compose_e2e as e2e  # noqa: E402
import load_test as lt  # noqa: E402
import pentest  # noqa: E402 - its Worker launcher and ways to reach the API

OUT = lt.ROOT / "data" / "rehearsal.json"
VISITOR_EVERY = 0.25


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)  # noqa: T201


class Visitor(threading.Thread):
    """A public page's request, through the Worker, four times a second."""

    def __init__(self) -> None:
        super().__init__(daemon=True)
        self.samples: list[tuple[float, str]] = []
        self.done = threading.Event()

    def run(self) -> None:
        with httpx.Client(base_url=f"{pentest.EDGE}/api", timeout=5) as client:
            while not self.done.is_set():
                at = time.monotonic()
                try:
                    status = str(client.get("/v1/gate/status").status_code)
                except httpx.HTTPError:
                    status = "no answer"
                self.samples.append((at, status))
                self.done.wait(VISITOR_EVERY)

    def between(self, start: float, end: float) -> dict[str, int]:
        """What visitors got in a window. 502 and "no answer" are the local restart; 403 isn't."""
        seen = Counter(status for at, status in self.samples if start <= at <= end)
        return dict(sorted(seen.items()))


class Rehearsal:
    def __init__(self) -> None:
        state = lt.load_state()
        self.state = state
        self.stack = lt.Stack(Path(state["env_file"]))
        self.env_file = Path(state["env_file"])
        values = lt.env_values(self.env_file)
        self.values = values
        self.edge_secret = state["compose_env"]["LT_EDGE_SECRET"]
        self.password = (lt.WORK / "user_password.txt").read_text("utf-8").strip()
        self.db_password = state["compose_env"].get(
            "LT_API_DB_PASSWORD", values["COMPOSE_LOGIN_PASSWORD"]
        )
        self.metrics_token = values["METRICS_BEARER_TOKEN"]
        self.worker: subprocess.Popen[bytes] | None = None
        self.visitor = Visitor()
        self.timeline: list[dict[str, Any]] = []
        self.started = time.monotonic()

    # --- The secret store and the deploys ---

    def store(self, **changes: str) -> None:
        """Change values on the "environment page". The hash's $ signs need single quotes."""
        lines = []
        for line in self.env_file.read_text("utf-8").splitlines():
            key = line.partition("=")[0]
            if key in changes:
                value = changes.pop(key)
                line = f"{key}='{value}'" if "$" in value else f"{key}={value}"
            lines.append(line)
        if changes:
            raise KeyError(f"not in the env file: {sorted(changes)}")
        e2e.write_private(self.env_file, "\n".join(lines) + "\n")

    def deploy(self, **switches: str) -> float:
        """Recreate the API with the store's values, as Render's deploy does. Seconds to ready."""
        return lt.start_api(self.stack, **switches)

    def restart_worker(self, secret: str) -> None:
        if self.worker is not None:
            self.worker.terminate()
            self.worker.wait(timeout=10)
        self.worker = pentest.start_edge(secret)

    def mark(self, step: str, **details: Any) -> float:
        now = time.monotonic()
        self.timeline.append({"at_s": round(now - self.started, 1), "step": step, **details})
        log(f"{step} {details if details else ''}")
        return now

    # --- Checks: does a value still open anything? ---

    def edge_status(self, secret: str) -> int:
        headers = {"x-edge-secret": secret, "x-wikiguard-client-ip": "10.40.0.1"}
        return httpx.get(f"{lt.API}/v1/gate/status", headers=headers, timeout=10).status_code

    def metrics_status(self, token: str) -> int:
        headers = lt.edge() | {"authorization": f"Bearer {token}"}
        return httpx.get(f"{lt.API}/metrics", headers=headers, timeout=10).status_code

    def login(self, password: str, address: str) -> tuple[int, str | None, httpx.Client | None]:
        client = pentest.Target(os.environ["LT_EDGE_SECRET"], password).as_visitor(address)
        tab = f"rehearse-{secrets.token_hex(5)}"
        reply = client.post(
            "/v1/auth/login", json={"password": password, "tab_id": tab, "take_over": True}
        )
        if reply.status_code != 200:
            return reply.status_code, reply.json().get("code"), None
        client.headers.update({"x-tab-id": tab, "x-csrf-token": reply.json()["csrf_token"]})
        return 200, None, client

    def owner_url(self) -> str:
        v = self.values
        return (
            f"postgresql://{v['POSTGRES_USER']}:{v['POSTGRES_PASSWORD']}"
            f"@127.0.0.1:{v['POSTGRES_PORT']}/{v['POSTGRES_DB']}"
        )

    def db_login_works(self, password: str) -> bool:
        v = self.values
        try:
            psycopg.connect(
                host="127.0.0.1",
                port=int(v["POSTGRES_PORT"]),
                dbname=v["POSTGRES_DB"],
                user="api_login",
                password=password,
                connect_timeout=5,
            ).close()
        except psycopg.OperationalError:
            return False
        return True

    def set_db_password(self, password: str) -> None:
        with psycopg.connect(self.owner_url(), autocommit=True) as connection:
            connection.execute(
                sql.SQL("ALTER ROLE api_login PASSWORD {}").format(sql.Literal(password))
            )

    def save_credentials(self) -> None:
        """Keep the files load_test.py and pentest.py read in step with the rotated values."""
        e2e.write_private(lt.WORK / "user_password.txt", self.password)
        self.state["compose_env"]["LT_EDGE_SECRET"] = self.edge_secret
        self.state["compose_env"]["LT_API_DB_PASSWORD"] = self.db_password
        self.state["metrics_token"] = self.metrics_token
        lt.STATE.write_text(json.dumps(self.state, indent=2), "utf-8")

    # --- 1. A scheduled rotation: every secret the API holds, with no 403 for visitors ---

    def scheduled_rotation(self) -> dict[str, Any]:
        old = {
            "edge": self.edge_secret,
            "password": self.password,
            "metrics": self.metrics_token,
            "db": self.db_password,
        }
        _, _, old_session = self.login(old["password"], "10.20.0.1")
        new = {
            "edge": secrets.token_urlsafe(32),
            "password": secrets.token_urlsafe(24),
            "metrics": secrets.token_urlsafe(24),
            "db": secrets.token_urlsafe(24),
        }
        start = self.mark("scheduled: start")
        self.set_db_password(new["db"])
        after_alter = self.mark("scheduled: api_login's password changed in Postgres")
        self.store(
            SESSION_SECRET_KEY=secrets.token_urlsafe(48),
            USER_PASSWORD_HASH=PasswordHasher().hash(new["password"]),
            METRICS_BEARER_TOKEN=new["metrics"],
        )
        ready_1 = self.deploy(
            LT_EDGE_SECRET=new["edge"],
            LT_EDGE_SECRET_PREVIOUS=old["edge"],
            LT_API_DB_PASSWORD=new["db"],
        )
        deployed_1 = self.mark("scheduled: deploy 1 ready (new values, old edge still accepted)")
        # The owner takes a moment before updating the Worker: visitors must see no 403 meanwhile.
        time.sleep(20)
        self.restart_worker(new["edge"])
        worker_done = self.mark("scheduled: Worker sends the new edge secret")
        ready_2 = self.deploy(LT_EDGE_SECRET_PREVIOUS="")
        end = self.mark("scheduled: deploy 2 ready (old edge secret dropped)")
        self.edge_secret, self.password = new["edge"], new["password"]
        self.metrics_token, self.db_password = new["metrics"], new["db"]
        self.save_credentials()
        # The session from before the rotation, sent with today's edge secret, so the answer is
        # about the session, not the door.
        if old_session is not None:
            old_session.headers["x-edge-secret"] = new["edge"]
        old_cookie = (
            old_session.get("/v1/queue").status_code if old_session is not None else "no session"
        )
        login_old = self.login(old["password"], "10.20.0.2")[:2]
        login_new = self.login(new["password"], "10.20.0.3")[0]
        checks = {
            "edge_old_secret": self.edge_status(old["edge"]),
            "edge_new_secret": self.edge_status(new["edge"]),
            "metrics_old_token": self.metrics_status(old["metrics"]),
            "metrics_new_token": self.metrics_status(new["metrics"]),
            "owner_old_password": login_old,
            "owner_new_password": login_new,
            "session_from_before": old_cookie,
            "api_login_old_password_connects": self.db_login_works(old["db"]),
            "api_login_new_password_connects": self.db_login_works(new["db"]),
        }
        held = (
            checks["edge_old_secret"] == 403
            and checks["edge_new_secret"] != 403
            and checks["metrics_old_token"] != 200
            and checks["metrics_new_token"] == 200
            and login_old[0] != 200
            and login_new == 200
            and old_cookie == 401
            and not checks["api_login_old_password_connects"]
            and checks["api_login_new_password_connects"]
        )
        windows = {
            "postgres_changed_to_deploy_1": self.visitor.between(after_alter, deployed_1),
            "deploy_1_to_worker": self.visitor.between(deployed_1, worker_done),
            "worker_to_deploy_2": self.visitor.between(worker_done, end),
            "whole_rotation": self.visitor.between(start, end),
        }
        return {
            "seconds": round(end - start, 1),
            "api_ready_seconds": [ready_1, ready_2],
            "visitors": windows,
            "checks": checks,
            "held": held,
        }

    # --- 2. An incident: the owner's password has leaked and someone used it ---

    def incident(self) -> dict[str, Any]:
        rev = int(
            self.stack.psql(
                "SELECT rev_id FROM edits WHERE rev_id NOT IN (SELECT rev_id FROM reviews) "
                "ORDER BY rev_id LIMIT 1"
            )
        )
        leaked = self.password
        attack_start = self.mark("incident: the attacker signs in with the leaked password")
        status, _, attacker = self.login(leaked, "198.51.100.7")
        if attacker is None:
            raise RuntimeError(f"the attacker's sign-in failed ({status})")
        attacker.put(f"/v1/edits/{rev}/review", json={"decision": "ok", "note": "rehearsal"})
        detect = self.mark("incident: the owner signs in and reads the audit log")
        _, _, owner = self.login(leaked, "10.20.0.1")
        if owner is None:
            raise RuntimeError("the owner couldn't sign in")
        entries = owner.get("/v1/admin/audit-log", params={"limit": 20}).json()["items"]
        found = [e for e in entries if e["action"] == "review.saved" and e["target"] == str(rev)]
        detected = self.mark("incident: found a review the owner didn't make", rev_id=rev)

        # Contain: one deploy that signs everyone out, changes the password, and stops writes.
        new_password = secrets.token_urlsafe(24)
        self.store(
            SESSION_SECRET_KEY=secrets.token_urlsafe(48),
            USER_PASSWORD_HASH=PasswordHasher().hash(new_password),
        )
        ready_contain = self.deploy(LT_READ_ONLY="true")
        contained = self.mark("incident: contained (read-only, everyone signed out)")
        self.password = new_password
        attacker_cookie = attacker.get("/v1/queue").status_code
        attacker_again = self.login(leaked, "198.51.100.8")[:2]
        _, _, owner = self.login(new_password, "10.20.0.1")
        write_while_read_only = (
            owner.put(f"/v1/edits/{rev}/review", json={"decision": "skip"}).json().get("code")
            if owner is not None
            else "owner could not sign in"
        )
        visitor_read = httpx.get(f"{pentest.EDGE}/api/v1/gate/status", timeout=10).status_code

        # Eradicate: the other secrets the API holds, then undo what the attacker changed.
        old_edge, old_metrics, old_db = self.edge_secret, self.metrics_token, self.db_password
        self.edge_secret = secrets.token_urlsafe(32)
        self.metrics_token = secrets.token_urlsafe(24)
        self.db_password = secrets.token_urlsafe(24)
        self.set_db_password(self.db_password)
        self.store(METRICS_BEARER_TOKEN=self.metrics_token)
        eradicate_start = self.mark(
            "incident: rotating the edge secret, metrics token, DB password"
        )
        ready_eradicate = self.deploy(
            LT_EDGE_SECRET=self.edge_secret,
            LT_EDGE_SECRET_PREVIOUS="",
            LT_API_DB_PASSWORD=self.db_password,
        )
        api_rotated = self.mark("incident: API deployed with the new secrets (old edge refused)")
        self.restart_worker(self.edge_secret)
        worker_done = self.mark("incident: Worker sends the new edge secret")
        # rev is an int read from the database, so the f-string can't carry SQL.
        self.stack.psql(f"DELETE FROM reviews WHERE rev_id = {rev}")  # noqa: S608 - an int
        removed = self.stack.psql(f"SELECT count(*) FROM reviews WHERE rev_id = {rev}") == "0"  # noqa: S608
        self.mark("incident: the attacker's review removed, as the owner login", rev_id=rev)

        # Recover.
        ready_recover = self.deploy(LT_READ_ONLY="false")
        recovered = self.mark("incident: recovered (writes back on)")
        self.save_credentials()
        _, _, owner = self.login(self.password, "10.20.0.1")
        write_after = (
            owner.put(f"/v1/edits/{rev}/review", json={"decision": "skip"}).status_code
            if owner is not None
            else 0
        )
        self.stack.psql(f"DELETE FROM reviews WHERE rev_id = {rev}")  # noqa: S608 - an int
        checks = {
            "audit_log_showed_the_review": bool(found),
            "attacker_cookie_after_containment": attacker_cookie,
            "attacker_password_after_containment": attacker_again,
            "owner_write_while_read_only": write_while_read_only,
            "visitor_read_while_read_only": visitor_read,
            "attackers_review_removed": removed,
            "owner_write_after_recovery": write_after,
            "edge_old_secret": self.edge_status(old_edge),
            "metrics_old_token": self.metrics_status(old_metrics),
            "api_login_old_password_connects": self.db_login_works(old_db),
        }
        held = (
            checks["audit_log_showed_the_review"]
            and attacker_cookie == 401
            and attacker_again[0] != 200
            and write_while_read_only == "read_only"
            and visitor_read == 200
            and checks["attackers_review_removed"]
            and write_after == 200
            and checks["edge_old_secret"] == 403
            and checks["metrics_old_token"] != 200
            and not checks["api_login_old_password_connects"]
        )
        return {
            "detect_to_contained_seconds": round(contained - detected, 1),
            "detect_to_recovered_seconds": round(recovered - detected, 1),
            "attack_to_detect_seconds": round(detect - attack_start, 1),
            "api_ready_seconds": [ready_contain, ready_eradicate, ready_recover],
            "visitors": {
                "containment_deploy": self.visitor.between(detected, contained),
                "eradication_api_to_worker": self.visitor.between(api_rotated, worker_done),
                "eradication_whole": self.visitor.between(eradicate_start, worker_done),
                "recovery_deploy": self.visitor.between(worker_done, recovered),
            },
            "checks": checks,
            "held": held,
        }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--only", choices=["scheduled", "incident"], help="run one part")
    args = parser.parse_args()
    rehearsal = Rehearsal()
    rehearsal.stack.psql("DELETE FROM active_session; DELETE FROM login_throttle")
    log("starting the API and the Worker")
    lt.start_api(rehearsal.stack, LT_CORS_ORIGINS=pentest.EDGE)
    rehearsal.restart_worker(rehearsal.edge_secret)
    rehearsal.visitor.start()
    time.sleep(5)
    result: dict[str, Any] = {"when": time.strftime("%Y-%m-%dT%H:%M:%S")}
    try:
        if args.only in (None, "scheduled"):
            result["scheduled_rotation"] = rehearsal.scheduled_rotation()
            rehearsal.stack.psql("DELETE FROM active_session; DELETE FROM login_throttle")
        if args.only in (None, "incident"):
            result["incident"] = rehearsal.incident()
    finally:
        rehearsal.visitor.done.set()
        if rehearsal.worker is not None:
            rehearsal.worker.terminate()
        result["timeline"] = rehearsal.timeline
        OUT.write_text(json.dumps(result, indent=2, default=str) + "\n", "utf-8")
    held = all(part["held"] for key, part in result.items() if isinstance(part, dict))
    log(f"{'held' if held else 'FAILED'}: {OUT}")
    return 0 if held else 1


if __name__ == "__main__":
    raise SystemExit(main())
