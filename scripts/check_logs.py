"""Check that no secret, session, username, or diff text reaches a log or a browser (PLAN.md,
Phase 8: "Verify that a secret or full diff never appears in logs").

    python scripts/check_logs.py stack       the whole pipeline in Compose, at DEBUG
    python scripts/load_test.py up
    python scripts/check_logs.py browser     the frontend and the real Worker, in Chromium
    python scripts/load_test.py down

stack: builds and runs every service (scripts/compose_e2e.py's stack and its full check: ingest,
score, label, sign in, review, predict, a retrain request, a replay, the archiver's flush), plus
the abuse a log is most likely to repeat: wrong passwords, forged cookies and CSRF tokens, a wrong
metrics token, broken bodies. Every log line from every container, Postgres and Redpanda included,
and the output of each one-off job, is then searched for:
  - every secret value the stack was given, the owner's password, the session cookies and CSRF
    tokens it issued, and the raw usernames from the recorded stream;
  - the text of every stored diff (any line of 30 characters or more from it).
LOG_LEVEL is DEBUG, which only the API honours; the workers and jobs always log at INFO (F-172).

browser: builds the frontend with every secret value set in the build's environment, as if the
build machine held them all, then runs the bundle check (`npm run check:bundle`) and
scripts/check_browser_secrets.mjs through the Worker. The Worker's and the Next.js server's own
output is searched as well.

Writes data/log_check.json. Findings name the secret and the service, never the value.
"""

import argparse
import json
import os
import re
import secrets
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent))
import compose_e2e as e2e  # noqa: E402
import load_test as lt  # noqa: E402
import pentest  # noqa: E402

ROOT = lt.ROOT
OUT = ROOT / "data" / "log_check.json"
FIXTURES = ROOT / "tests" / "fixtures" / "wikimedia"
SECRET_KEYS = (
    "POSTGRES_PASSWORD",
    "COMPOSE_LOGIN_PASSWORD",
    "USER_HASH_KEY",
    "SESSION_SECRET_KEY",
    "USER_PASSWORD_HASH",
    "METRICS_BEARER_TOKEN",
)
DIFF_PIECE = 30


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)  # noqa: T201


class RecordingStack(e2e.Stack):
    """compose_e2e's stack, keeping both streams of every command, which is where one-off jobs
    (the labeler, the retrain poller) log: their containers are removed when they finish."""

    def __init__(self, env_file: Path) -> None:
        super().__init__(env_file)
        self.outputs: list[tuple[str, str]] = []

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
        if "run" in args[:4]:
            self.outputs.append((" ".join(args[:6]), done.stdout + done.stderr))
        if check and done.returncode != 0:
            print(done.stdout[-3000:], done.stderr[-3000:], file=sys.stderr)  # noqa: T201
            raise RuntimeError(f"docker compose {args[0]} failed")
        return done.stdout


def secret_values(env_file: Path, extra: dict[str, str]) -> dict[str, str]:
    values = lt.env_values(env_file)
    found = {key: values[key] for key in SECRET_KEYS if values.get(key)}
    # The Argon2 hash's own digest, in case a log quotes it without its parameters.
    found["USER_PASSWORD_HASH digest"] = found["USER_PASSWORD_HASH"].rsplit("$", 1)[-1]
    return found | extra


def fixture_usernames() -> set[str]:
    """Raw usernames in the recorded stream: stored only as a keyed hash, never logged."""
    names: set[str] = set()
    for path in FIXTURES.rglob("*"):
        if path.is_file():
            for name in re.findall(
                r'"user"\s*:\s*"([^"]{4,})"', path.read_text("utf-8", "replace")
            ):
                names.add(name)
    return names


def diff_pieces(stack: e2e.Stack) -> set[str]:
    """Each stored diff's lines of DIFF_PIECE characters or more, cut to that length."""
    text = stack.psql(
        "SELECT string_agg(coalesce(added_text, '') || E'\\n' || coalesce(removed_text, ''), "
        "E'\\n') FROM edits"
    )
    pieces = set()
    for line in text.splitlines():
        line = line.strip()
        if len(line) >= DIFF_PIECE:
            pieces.add(line[:DIFF_PIECE])
    return pieces


def search(sources: dict[str, str], needles: dict[str, str]) -> list[str]:
    """Which needle (by name) shows up in which source. Values are never returned."""
    hits = []
    for source, text in sources.items():
        for name, value in needles.items():
            if value and value in text:
                hits.append(f"{name} in {source}")
    return sorted(hits)


def api_ready() -> bool:
    try:
        return httpx.get(f"http://127.0.0.1:{e2e.API_PORT}/ready", timeout=5).status_code == 200
    except httpx.HTTPError:
        return False


def abuse(user_password: str) -> dict[str, str]:
    """What a log is most likely to repeat: bad credentials, forged tokens, broken bodies. Returns
    the session cookie and CSRF token a good sign-in got, to look for afterwards."""
    client = httpx.Client(base_url=f"http://127.0.0.1:{e2e.API_PORT}", timeout=30)
    tab = "tab-logcheck-aaaa"
    for attempt in range(3):
        client.post("/v1/auth/login", json={"password": f"wrong-{attempt}", "tab_id": tab})
    client.post("/v1/auth/login", json={"password": user_password[::-1], "tab_id": tab})
    login = client.post(
        "/v1/auth/login", json={"password": user_password, "tab_id": tab, "take_over": True}
    )
    login.raise_for_status()
    csrf = login.json()["csrf_token"]
    cookie = client.cookies.get("wikiguard_session") or ""
    headers = {"x-tab-id": tab}
    client.put("/v1/edits/1/review", headers=headers | {"x-csrf-token": "forged"}, json={})
    client.put(
        "/v1/edits/1/review",
        headers=headers | {"x-csrf-token": csrf},
        content=b'{"decision": "ok", "note": ',
    )
    client.post("/v1/predict", headers=headers | {"x-csrf-token": csrf}, json={"added_text": 5})
    httpx.get(
        f"http://127.0.0.1:{e2e.API_PORT}/v1/queue",
        cookies={"wikiguard_session": "forged-" + secrets.token_urlsafe(16)},
        timeout=10,
    )
    httpx.get(
        f"http://127.0.0.1:{e2e.API_PORT}/metrics",
        headers={"authorization": "Bearer wrong-token"},
        timeout=10,
    )
    client.post("/v1/auth/logout", headers=headers | {"x-csrf-token": csrf})
    return {"session cookie": cookie, "CSRF token": csrf}


def check_stack() -> dict[str, Any]:
    folder = Path(tempfile.mkdtemp(prefix="wikiguard-logcheck-"))
    env_file, user_password = e2e.write_env(folder)
    text = env_file.read_text("utf-8").replace("LOG_LEVEL=INFO", "LOG_LEVEL=DEBUG")
    e2e.write_private(env_file, text)
    stack = RecordingStack(env_file)
    try:
        stack.clean()
        log("building and starting every service, at DEBUG")
        stack.run("build", timeout=1800)
        stack.run("up", "-d", timeout=600)
        log("running the end-to-end check")
        pipeline = e2e.check_pipeline(stack, user_password)
        log("abusing the API")
        stack.run("start", "api", timeout=120)
        e2e.wait_for("the API to restart", api_ready, e2e.WAIT_READY_SECONDS)
        issued = abuse(user_password)
        stack.run("--profile", "jobs", "run", "--rm", "labeler", timeout=300)
        logs = stack.run(
            "--profile", "jobs", "logs", "--no-color", "--timestamps", check=False, timeout=300
        )
        # Kept for reading by eye: it holds only this run's throwaway values (data/ is ignored).
        e2e.write_private(lt.WORK / "log_check_raw.txt", logs)
        by_service: dict[str, list[str]] = {}
        for line in logs.splitlines():
            service = line.split("|", 1)[0].strip() or "unknown"
            by_service.setdefault(service, []).append(line)
        sources = {name: "\n".join(lines) for name, lines in by_service.items()}
        for index, (command, output) in enumerate(stack.outputs):
            sources[f"job run {index}: {command}"] = output
        needles = secret_values(env_file, {"owner password": user_password} | issued)
        names = {f"username #{i}": name for i, name in enumerate(sorted(fixture_usernames()))}
        diffs = {f"diff piece #{i}": piece for i, piece in enumerate(sorted(diff_pieces(stack)))}
        return {
            "pipeline": pipeline,
            "log_lines": len(logs.splitlines()),
            "lines_by_source": {name: text.count("\n") + 1 for name, text in sources.items()},
            "secrets_checked": len(needles),
            "usernames_checked": len(names),
            "diff_pieces_checked": len(diffs),
            "found": search(sources, needles | names | diffs),
        }
    finally:
        stack.clean()
        env_file.unlink(missing_ok=True)


def frontend_up() -> bool:
    try:
        return httpx.get(f"{pentest.EDGE}/", timeout=10).status_code == 200
    except httpx.HTTPError:
        return False


def check_browser(skip_build: bool) -> dict[str, Any]:
    state = lt.load_state()
    env_file = Path(state["env_file"])
    edge_secret = state["compose_env"]["LT_EDGE_SECRET"]
    password_file = lt.WORK / "user_password.txt"
    password = password_file.read_text("utf-8").strip()
    values = secret_values(
        env_file, {"EDGE_SHARED_SECRET": edge_secret, "owner password": password}
    )
    # The build and the server get every secret, under its real name, as a careless build machine
    # might. Plus two in the shapes the bundle check knows, so its pattern search is exercised too.
    fake_openai = "sk-proj-" + secrets.token_urlsafe(40).replace("-", "a").replace("_", "b")
    build_env = {key: os.environ[key] for key in os.environ if key != "API_ORIGIN"} | {
        "SESSION_SECRET_KEY": values["SESSION_SECRET_KEY"],
        "USER_PASSWORD_HASH": values["USER_PASSWORD_HASH"],
        "USER_HASH_KEY": values["USER_HASH_KEY"],
        "METRICS_BEARER_TOKEN": values["METRICS_BEARER_TOKEN"],
        "EDGE_SHARED_SECRET": edge_secret,
        "DATABASE_URL": f"postgresql://api_login:{values['COMPOSE_LOGIN_PASSWORD']}@db:5432/x",
        "OPENAI_API_KEY": fake_openai,
    }
    values["fake OpenAI key"] = fake_openai
    npm = "npm.cmd" if os.name == "nt" else "npm"
    frontend = ROOT / "frontend"
    result: dict[str, Any] = {}
    if not skip_build:
        log("building the frontend with every secret in its environment")
        subprocess.run(  # noqa: S603 - fixed command
            [npm, "run", "build"], cwd=frontend, env=build_env, check=True, capture_output=True
        )
    bundle = subprocess.run(  # noqa: S603 - fixed command
        [npm, "run", "check:bundle"],
        cwd=frontend,
        env=build_env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    result["bundle_check_passed"] = bundle.returncode == 0
    result["bundle_check_summary"] = [
        line for line in bundle.stdout.splitlines() if line.startswith("Secrets")
    ]
    lt.start_api(lt.Stack(env_file), LT_CORS_ORIGINS=pentest.EDGE)
    lt.Stack(env_file).psql("DELETE FROM active_session; DELETE FROM login_throttle")
    edit = lt.Stack(env_file).psql("SELECT rev_id FROM edits ORDER BY latest_score DESC LIMIT 1")
    server_log = tempfile.NamedTemporaryFile("w+", delete=False, suffix=".log")  # noqa: SIM115
    worker_log = tempfile.NamedTemporaryFile("w+", delete=False, suffix=".log")  # noqa: SIM115
    server = subprocess.Popen(  # noqa: S603 - fixed command
        [npm, "start"], cwd=frontend, env=build_env, stdout=server_log, stderr=server_log
    )
    worker = subprocess.Popen(  # noqa: S603 - fixed command
        ["node", "scripts/edge_local.mjs", "--api", lt.API],  # noqa: S607 - node is on PATH
        cwd=ROOT,
        env=os.environ | {"EDGE_SHARED_SECRET": edge_secret},
        stdout=worker_log,
        stderr=worker_log,
    )
    secrets_file = lt.WORK / "secret_values.json"
    e2e.write_private(secrets_file, json.dumps(list(values.values())))
    try:
        e2e.wait_for("the frontend", frontend_up, 120)
        done = subprocess.run(  # noqa: S603 - fixed command
            ["node", "scripts/check_browser_secrets.mjs", "--edit", edit],  # noqa: S607
            cwd=ROOT,
            env=os.environ
            | {"PASSWORD_FILE": str(password_file), "SECRETS_FILE": str(secrets_file)},
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
        line = next((x for x in done.stdout.splitlines() if x.startswith("RESULT ")), None)
        if line is None:
            raise RuntimeError(f"the browser check failed: {done.stderr[-1500:]}")
        result["browser"] = json.loads(line.removeprefix("RESULT "))
    finally:
        secrets_file.unlink(missing_ok=True)
        if os.name == "nt":
            # Before terminate(): npm's child, the server itself, is only found through its parent.
            subprocess.run(  # noqa: S603 - fixed command
                ["taskkill", "/F", "/T", "/PID", str(server.pid)],  # noqa: S607
                capture_output=True,
                check=False,
            )
        server.terminate()
        worker.terminate()
    for process in (server, worker):
        try:
            process.wait(timeout=20)
        except subprocess.TimeoutExpired:
            process.kill()
    outputs = {
        "Next.js server output": Path(server_log.name).read_text("utf-8", "replace"),
        "Worker output": Path(worker_log.name).read_text("utf-8", "replace"),
    }
    server_log.close()
    worker_log.close()
    for path in (server_log.name, worker_log.name):
        try:
            Path(path).unlink(missing_ok=True)
        except PermissionError:  # Windows: a child of npm may still hold it for a moment
            time.sleep(3)
            Path(path).unlink(missing_ok=True)
    result["found_in_server_output"] = search(outputs, values)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("part", choices=["stack", "browser"])
    parser.add_argument("--skip-build", action="store_true", help="browser: use the last build")
    args = parser.parse_args()
    previous = json.loads(OUT.read_text("utf-8")) if OUT.exists() else {}
    result = check_stack() if args.part == "stack" else check_browser(args.skip_build)
    previous[args.part] = {"when": time.strftime("%Y-%m-%dT%H:%M:%S")} | result
    OUT.write_text(json.dumps(previous, indent=2, default=str) + "\n", "utf-8")
    if args.part == "stack":
        clean = not result["found"]
    else:
        seen = result["browser"]
        clean = (
            result["bundle_check_passed"]
            and not result["found_in_server_output"]
            and not seen["found_in_responses"]
            and not seen["foreign_origins"]
            and seen["session_cookie"] is not None
            and seen["session_cookie"]["http_only"]
            and not seen["session_cookie_in_document_cookie"]
            and not seen["session_value_in_storage"]
            and seen["csrf_token_seen"]
            and not seen["csrf_token_in_storage_or_cookie"]
        )
    log(f"{'clean' if clean else 'FOUND SOMETHING'}: {OUT}")
    return 0 if clean else 1


if __name__ == "__main__":
    raise SystemExit(main())
