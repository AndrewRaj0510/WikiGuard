"""A stand-in for OpenAI and Gemini, for the Explain load test (scripts/load_test.py, Phase 8).

Tests must not call an LLM (CLAUDE.md). On the load-test Compose network this container answers to
api.openai.com and generativelanguage.googleapis.com, over TLS with a throwaway certificate the
script makes for the run; the API trusts it through SSL_CERT_FILE, so its code is unchanged. Each
answer comes after FAKE_LLM_DELAY_MS, about what a small hosted model takes for one sentence.

    :443   POST /v1/chat/completions                 (OpenAI's shape)
           POST /v1beta/models/NAME:generateContent  (Gemini's shape)
    :8080  GET  /stats                               calls so far, the most at once, and the
                                                     last few prompts it was sent
           POST /answer                              what to answer from now on (the body)

FAKE_LLM_ANSWER sets the answer from the start; the penetration test uses /answer to make it
return markup or a link, the way a manipulated model might.

Standard library only.
"""

import json
import os
import ssl
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

CERT_DIR = sys.argv[1] if len(sys.argv) > 1 else "/llm-cert"
DELAY_SECONDS = int(os.environ.get("FAKE_LLM_DELAY_MS", "1500")) / 1000
ANSWER = os.environ.get("FAKE_LLM_ANSWER") or (
    "This edit adds a short sentence to the article and changes no other text."
)
KEEP_PROMPTS = 5

lock = threading.Lock()
stats = {"openai": 0, "gemini": 0, "in_flight": 0, "most_in_flight": 0, "prompts": []}
answer = {"text": ANSWER}


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, format: str, *args: object) -> None:
        return

    def _json(self, body: dict, status: int = 200) -> None:
        data = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:
        if self.path == "/stats":
            with lock:
                self._json(dict(stats))
        else:
            self.send_error(404)

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length)
        if self.path == "/answer":
            with lock:
                answer["text"] = raw.decode("utf-8")
            self._json({"ok": True})
            return
        with lock:
            text = answer["text"]
        if self.path.endswith("/chat/completions"):
            provider, body = "openai", {"choices": [{"message": {"content": text}}]}
        elif ":generateContent" in self.path:
            provider = "gemini"
            body = {"candidates": [{"content": {"parts": [{"text": text}]}}]}
        else:
            self.send_error(404)
            return
        with lock:
            stats[provider] += 1
            stats["prompts"] = [*stats["prompts"], json.loads(raw or b"{}")][-KEEP_PROMPTS:]
            stats["in_flight"] += 1
            stats["most_in_flight"] = max(stats["most_in_flight"], stats["in_flight"])
        try:
            time.sleep(DELAY_SECONDS)
            self._json(body)
        finally:
            with lock:
                stats["in_flight"] -= 1


def serve_tls() -> None:
    server = ThreadingHTTPServer(("0.0.0.0", 443), Handler)  # noqa: S104 - in a container
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    # Refuse TLS 1.0 and 1.1 even in a test fake: the API's client never needs them (code scanning).
    context.minimum_version = ssl.TLSVersion.TLSv1_2
    context.load_cert_chain(f"{CERT_DIR}/cert.pem", f"{CERT_DIR}/key.pem")
    server.socket = context.wrap_socket(server.socket, server_side=True)
    server.serve_forever()


if __name__ == "__main__":
    threading.Thread(target=serve_tls, daemon=True).start()
    print(f"Fake LLM on :443, {DELAY_SECONDS} s per answer; stats on :8080", flush=True)
    ThreadingHTTPServer(("0.0.0.0", 8080), Handler).serve_forever()  # noqa: S104 - in a container
