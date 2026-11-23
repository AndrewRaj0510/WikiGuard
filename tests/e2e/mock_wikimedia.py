"""A stand-in for Wikimedia, serving the recorded fixtures, so the end-to-end run touches no wiki.

Tests must not call Wikimedia (CLAUDE.md), and this lets the real ingestor and labeler run
unchanged: only their two URL settings point here. It serves:

    /stream     the recorded in-scope stream events as SSE, then keeps the connection open and
                sends any event added later. A reconnect with Last-Event-ID gets every event after
                that id, as EventStreams does, which is how the ingestor repairs a gap.
    /w/api.php  compare, revisions, and users, in the shapes the recorded responses have

Half the edits are dated 49 hours ago, so the labeler can give them final labels now. Every fourth
of those is tagged mw-reverted, so both labels appear. Run with the standard library only.

For failure injection (scripts/inject_failures.py), three more paths. Without them nothing changes:

    /control?stream=down|up         down drops open streams and answers 503 until up
    /control?api=429|maxlag&count=N the next N API calls fail that way, with Retry-After: WAIT
            &wait=SECONDS
    /control?add=N                  add N new edits, dated now, to the stream
    /stats                          the edit count, and each API call and stream connection
"""

import json
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

FIXTURES = Path(sys.argv[1] if len(sys.argv) > 1 else "tests/fixtures/wikimedia")
PORT = 8080
OLD_AGE_SECONDS = 49 * 3600
RECENT_AGE_SECONDS = 300
COMPARE_CASES = ["small_changed", "added_only", "removed_only", "multi_section", "unicode"]
# Added edits get revision ids from here up, far from the recorded ones.
ADDED_REV_START = 9_000_000_000


def in_scope_events() -> list[dict]:
    events = []
    with (FIXTURES / "stream_events.jsonl").open(encoding="utf-8") as file:
        for line in file:
            row = json.loads(line)
            event = row["event"]
            if (
                event.get("type") == "edit"
                and event.get("namespace") == 0
                and event.get("bot") is False
            ):
                # The recorded events are from another wiki or category too, so all become enwiki.
                event["wiki"] = "enwiki"
                event["server_name"] = "en.wikipedia.org"
                events.append(event)
    return events


EVENTS = in_scope_events()
COMPARES = [
    json.loads((FIXTURES / f"compare_{case}.json").read_text("utf-8"))["response"]
    for case in COMPARE_CASES
]
AGED = {event["revision"]["new"] for index, event in enumerate(EVENTS) if index % 2 == 0}
REVERTED = {event["revision"]["new"] for index, event in enumerate(EVENTS) if index % 8 == 0}


class State:
    """What the control path changes. One lock, because the server answers on many threads."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.started = int(time.time())
        # (body, timestamp) for each event, in stream order. The list index is the SSE id.
        self.log: list[dict] = []
        for event in EVENTS:
            age = OLD_AGE_SECONDS if event["revision"]["new"] in AGED else RECENT_AGE_SECONDS
            self.log.append(event | {"timestamp": self.started - age})
        self.served = False
        self.stream_down = False
        self.stream_generation = 0
        self.api_failure: str | None = None
        self.api_failures_left = 0
        self.api_wait = 0
        self.api_calls: list[dict] = []
        self.stream_connects: list[dict] = []
        self.added = 0

    def add(self, count: int) -> list[int]:
        with self.lock:
            rev_ids = []
            for _ in range(count):
                template = EVENTS[self.added % len(EVENTS)]
                rev_id = ADDED_REV_START + self.added
                self.added += 1
                body = template | {
                    "timestamp": int(time.time()),
                    "revision": {"old": rev_id - 1, "new": rev_id},
                    "title": f"Injected article {self.added}",
                }
                self.log.append(body)
                rev_ids.append(rev_id)
            return rev_ids


STATE = State()


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, format: str, *args: object) -> None:
        return

    def _json(self, body: dict, status: int = 200, headers: dict[str, str] | None = None) -> None:
        data = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        for key, value in (headers or {}).items():
            self.send_header(key, value)
        self.end_headers()
        self.wfile.write(data)

    def _stream(self) -> None:
        last = self.headers.get("Last-Event-ID")
        with STATE.lock:
            STATE.stream_connects.append(
                {
                    "at": time.time(),
                    "answer": 503 if STATE.stream_down else 200,
                    "last_event_id": last,
                }
            )
            if STATE.stream_down:
                down = True
            else:
                down = False
                generation = STATE.stream_generation
                if last is not None and last.isdigit():
                    position = int(last) + 1
                elif not STATE.served:
                    position = 0
                else:
                    # A fresh connection with no position starts at "now", as the real one does.
                    position = len(STATE.log)
                STATE.served = True
        if down:
            self._json({"error": "stream down (injected)"}, status=503)
            return
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Connection", "close")
        self.end_headers()
        last_write = 0.0
        try:
            while True:
                with STATE.lock:
                    if STATE.stream_down or STATE.stream_generation != generation:
                        return
                    pending = STATE.log[position:]
                for offset, body in enumerate(pending):
                    message = (
                        f"id: {position + offset}\nevent: message\ndata: {json.dumps(body)}\n\n"
                    )
                    self.wfile.write(message.encode())
                position += len(pending)
                if pending or time.monotonic() - last_write >= 5:
                    if not pending:
                        self.wfile.write(b": keep-alive\n\n")
                    self.wfile.flush()
                    last_write = time.monotonic()
                time.sleep(0.2)
        except OSError:
            return

    def _revisions(self, ids: list[int]) -> dict:
        revisions = []
        for rev_id in ids:
            tags = ["mw-reverted"] if rev_id in REVERTED else []
            revisions.append({"revid": rev_id, "parentid": rev_id - 1, "tags": tags})
        return {"query": {"pages": [{"pageid": 1, "ns": 0, "revisions": revisions}]}}

    def _control(self, params: dict[str, str]) -> None:
        answer: dict = {}
        with STATE.lock:
            if params.get("stream") == "down":
                STATE.stream_down = True
                STATE.stream_generation += 1
            elif params.get("stream") == "up":
                STATE.stream_down = False
            if params.get("api") in {"429", "maxlag"}:
                STATE.api_failure = params["api"]
                STATE.api_failures_left = int(params.get("count", "1"))
                STATE.api_wait = int(params.get("wait", "2"))
            elif params.get("api") == "ok":
                STATE.api_failure, STATE.api_failures_left = None, 0
        if "add" in params:
            answer["added"] = STATE.add(int(params["add"]))
        self._json(answer | {"ok": True})

    def _stats(self) -> None:
        with STATE.lock:
            body = {
                "events": len(STATE.log),
                "added": STATE.added,
                "stream_down": STATE.stream_down,
                "api_calls": STATE.api_calls[-500:],
                "stream_connects": STATE.stream_connects[-500:],
            }
        self._json(body)

    def do_GET(self) -> None:
        url = urlparse(self.path)
        params = {key: values[0] for key, values in parse_qs(url.query).items()}
        if url.path == "/health":
            self._json({"status": "ok"})
        elif url.path == "/stream":
            self._stream()
        elif url.path == "/w/api.php":
            self._api(params)
        elif url.path == "/control":
            self._control(params)
        elif url.path == "/stats":
            self._stats()
        else:
            self.send_error(404)

    def _injected_failure(self) -> str | None:
        with STATE.lock:
            if STATE.api_failures_left <= 0:
                return None
            STATE.api_failures_left -= 1
            return STATE.api_failure

    def _api(self, params: dict[str, str]) -> None:
        failure = self._injected_failure()
        with STATE.lock:
            STATE.api_calls.append(
                {
                    "at": time.time(),
                    "action": params.get("action") or params.get("prop") or params.get("list"),
                    "answer": failure or "ok",
                    "maxlag": params.get("maxlag"),
                }
            )
            wait = str(STATE.api_wait)
        if failure == "429":
            self._json({"error": "too many requests"}, status=429, headers={"Retry-After": wait})
            return
        if failure == "maxlag":
            # The real API sends this inside HTTP 200 (memory: "API errors arrive inside 200").
            body = {"error": {"code": "maxlag", "info": "Waiting for a database server: 6 s lag"}}
            self._json(body, headers={"Retry-After": wait})
            return
        if params.get("action") == "compare":
            case = COMPARES[int(params.get("torev", "0")) % len(COMPARES)]
            self._json(case)
        elif params.get("prop") == "revisions":
            self._json(self._revisions([int(item) for item in params["revids"].split("|")]))
        elif params.get("list") == "users":
            names = params.get("ususers", "").split("|")
            users = [{"name": name, "editcount": 42} for name in names if name]
            self._json({"query": {"users": users}})
        else:
            self._json({"error": {"code": "unknown_request"}})

    do_POST = do_GET


if __name__ == "__main__":
    print(f"Serving {len(EVENTS)} recorded edits on {PORT}", flush=True)
    ThreadingHTTPServer(("0.0.0.0", PORT), Handler).serve_forever()  # noqa: S104 - in a container
