"""A few fast checks that a deployed API is the new build, is up, and is still locked.

Run by the deploy workflow against staging, and again against production. It only reads, signs in
as nobody, and takes no seat, so running it against production changes nothing.

    python scripts/smoke_test.py --api https://staging-api.example --commit 0123abc
    python scripts/smoke_test.py --api https://site.example/api --origin https://api.onrender.com

--api is where the API answers: the Render address for staging, or the site's /api path for
production, where the Worker sits in front. --origin is the Render address behind a Worker, to
check that it refuses requests that skip the Worker.

A free Render service sleeps, and a deploy takes minutes, so the first check waits (--wait).
"""

import argparse
import sys
import time
from collections.abc import Callable
from typing import Any

import httpx

DEFAULT_WAIT_SECONDS = 900
POLL_SECONDS = 10
REQUIRED_HEADERS = {
    "x-content-type-options": "nosniff",
    "referrer-policy": "no-referrer",
    "x-frame-options": "DENY",
}
PROTECTED_PATH = "/v1/queue"
# Every console request names its browser tab. Without this header the API answers 400 before it
# looks for a session, and the check below is about the session.
TAB_HEADERS = {"x-tab-id": "tab-smoketest"}
# Words that only appear when a server leaks its internals in an error.
LEAK_MARKERS = ("Traceback", "psycopg", "sqlalchemy", 'File "')


class SmokeFailure(Exception):
    pass


def expect(condition: bool, message: str) -> None:
    if not condition:
        raise SmokeFailure(message)


def wait_until(
    check: Callable[[], str | None],
    seconds: float,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
) -> None:
    """Calls check until it returns None. Anything else is the reason it isn't ready yet."""
    deadline = clock() + seconds
    while True:
        reason = check()
        if reason is None:
            return
        if clock() >= deadline:
            raise SmokeFailure(f"gave up after {seconds:.0f} s: {reason}")
        sleep(POLL_SECONDS)


def same_commit(served: str, wanted: str) -> bool:
    """Either may be a short form of the other, and seven characters is git's shortest."""
    shortest = min(len(served), len(wanted))
    return shortest >= 7 and served[:shortest] == wanted[:shortest]


def check_new_build(client: httpx.Client, commit: str | None) -> Callable[[], str | None]:
    def check() -> str | None:
        try:
            response = client.get("/health")
        except httpx.HTTPError as error:
            return f"/health didn't answer ({type(error).__name__})"
        if response.status_code != 200:
            return f"/health returned {response.status_code}"
        served = str(response.json().get("commit", ""))
        if commit and not same_commit(served, commit):
            return f"/health is still serving commit {served or 'unknown'}"
        return None

    return check


def check_ready(client: httpx.Client) -> Callable[[], str | None]:
    def check() -> str | None:
        try:
            response = client.get("/ready")
        except httpx.HTTPError as error:
            return f"/ready didn't answer ({type(error).__name__})"
        if response.status_code != 200:
            return f"/ready returned {response.status_code}: {response.text[:200]}"
        return None

    return check


def check_locked(client: httpx.Client) -> None:
    """The controls that matter most still hold on the deployed build."""
    queue = client.get(PROTECTED_PATH, headers=TAB_HEADERS)
    expect(queue.status_code == 401, f"{PROTECTED_PATH} without a session: {queue.status_code}")
    expect(
        "problem+json" in queue.headers.get("content-type", ""),
        "errors aren't in the problem format",
    )
    metrics = client.get("/metrics")
    expect(metrics.status_code == 401, f"/metrics without a token returned {metrics.status_code}")

    schema = client.get("/v1/openapi.json")
    expect(schema.status_code == 200, f"/v1/openapi.json returned {schema.status_code}")
    paths: dict[str, Any] = schema.json().get("paths", {})
    expect(PROTECTED_PATH in paths, "the public schema is missing the queue")
    expect(not [path for path in paths if "admin" in path], "the public schema lists admin routes")

    missing = client.get("/v1/no-such-route")
    expect(missing.status_code == 404, f"an unknown route returned {missing.status_code}")
    expect(
        not any(marker in missing.text for marker in LEAK_MARKERS),
        "an error body leaks internals",
    )
    for name, value in REQUIRED_HEADERS.items():
        expect(missing.headers.get(name) == value, f"header {name} is missing or wrong")
    expect("x-request-id" in missing.headers, "responses carry no request ID")
    expect(
        "frame-ancestors 'none'" in missing.headers.get("content-security-policy", ""),
        "the content security policy doesn't forbid framing",
    )


def check_origin_closed(origin: httpx.Client) -> None:
    """Behind a Worker, only /health may be reached directly."""
    health = origin.get("/health")
    expect(health.status_code == 200, f"the origin's /health returned {health.status_code}")
    direct = origin.get("/ready")
    expect(direct.status_code == 403, f"a direct request to the origin got {direct.status_code}")


def run(
    api: httpx.Client,
    origin: httpx.Client | None,
    commit: str | None,
    wait_seconds: float,
    sleep: Callable[[float], None] = time.sleep,
) -> list[str]:
    done = []
    wait_until(check_new_build(api, commit), wait_seconds, sleep)
    done.append("the new build is serving" if commit else "the API answers")
    wait_until(check_ready(api), wait_seconds, sleep)
    done.append("ready: database reachable and model loaded")
    check_locked(api)
    done.append("still locked: sessions, metrics token, hidden admin routes, headers")
    if origin is not None:
        check_origin_closed(origin)
        done.append("the origin refuses requests that skip the Worker")
    return done


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--api", required=True, help="where the API answers")
    parser.add_argument("--origin", help="the address behind the Worker, which must refuse us")
    parser.add_argument("--commit", help="wait until /health reports this commit")
    parser.add_argument("--wait", type=float, default=DEFAULT_WAIT_SECONDS)
    args = parser.parse_args()
    headers = {"user-agent": "WikiGuard-smoke-test/1.0"}
    with httpx.Client(base_url=args.api.rstrip("/"), timeout=60, headers=headers) as api:
        origin = (
            httpx.Client(base_url=args.origin.rstrip("/"), timeout=60, headers=headers)
            if args.origin
            else None
        )
        try:
            for line in run(api, origin, args.commit, args.wait):
                print(f"ok: {line}")  # noqa: T201
        except SmokeFailure as failure:
            print(f"FAILED: {failure}", file=sys.stderr)  # noqa: T201
            return 1
        finally:
            if origin is not None:
                origin.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
