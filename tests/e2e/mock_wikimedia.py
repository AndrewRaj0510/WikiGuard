"""A stand-in for Wikimedia, serving the recorded fixtures, so the end-to-end run touches no wiki.

Tests must not call Wikimedia (CLAUDE.md), and this lets the real ingestor and labeler run
unchanged: only their two URL settings point here. It serves:

    /stream     the recorded in-scope stream events as SSE, once, then keeps the connection open
    /w/api.php  compare, revisions, and users, in the shapes the recorded responses have

Half the edits are dated 49 hours ago, so the labeler can give them final labels now. Every fourth
of those is tagged mw-reverted, so both labels appear. Run with the standard library only.
"""

import json
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

FIXTURES = Path(sys.argv[1] if len(sys.argv) > 1 else "tests/fixtures/wikimedia")
PORT = 8080
OLD_AGE_SECONDS = 49 * 3600
RECENT_AGE_SECONDS = 300
COMPARE_CASES = ["small_changed", "added_only", "removed_only", "multi_section", "unicode"]


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
served = False


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, format: str, *args: object) -> None:
        return

    def _json(self, body: dict) -> None:
        data = json.dumps(body).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _stream(self) -> None:
        global served
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Connection", "close")
        self.end_headers()
        now = int(time.time())
        if not served:
            served = True
            for index, event in enumerate(EVENTS):
                age = OLD_AGE_SECONDS if event["revision"]["new"] in AGED else RECENT_AGE_SECONDS
                body = event | {"timestamp": now - age}
                message = f"id: {index}\nevent: message\ndata: {json.dumps(body)}\n\n"
                self.wfile.write(message.encode())
            self.wfile.flush()
        try:
            while True:
                self.wfile.write(b": keep-alive\n\n")
                self.wfile.flush()
                time.sleep(5)
        except OSError:
            return

    def _revisions(self, ids: list[int]) -> dict:
        revisions = []
        for rev_id in ids:
            tags = ["mw-reverted"] if rev_id in REVERTED else []
            revisions.append({"revid": rev_id, "parentid": rev_id - 1, "tags": tags})
        return {"query": {"pages": [{"pageid": 1, "ns": 0, "revisions": revisions}]}}

    def do_GET(self) -> None:
        url = urlparse(self.path)
        params = {key: values[0] for key, values in parse_qs(url.query).items()}
        if url.path == "/health":
            self._json({"status": "ok"})
        elif url.path == "/stream":
            self._stream()
        elif url.path == "/w/api.php":
            self._api(params)
        else:
            self.send_error(404)

    def _api(self, params: dict[str, str]) -> None:
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
