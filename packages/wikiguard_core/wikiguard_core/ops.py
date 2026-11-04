"""GET /health and GET /metrics for workers, which have no web framework of their own."""

import threading
from collections.abc import Callable
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from prometheus_client import CONTENT_TYPE_LATEST, CollectorRegistry, generate_latest


def serve_ops(
    port: int,
    registry: CollectorRegistry,
    is_healthy: Callable[[], bool],
    host: str = "0.0.0.0",  # noqa: S104 - in a pod; NetworkPolicies limit who can reach it
) -> ThreadingHTTPServer:
    """Start the endpoints on a daemon thread, so they never hold up the worker."""

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            if self.path == "/health":
                healthy = is_healthy()
                self._send(
                    200 if healthy else 503,
                    "application/json",
                    b'{"status":"%b"}' % (b"ok" if healthy else b"unhealthy"),
                )
            elif self.path == "/metrics":
                self._send(200, CONTENT_TYPE_LATEST, generate_latest(registry))
            else:
                self._send(404, "text/plain", b"not found")

        def _send(self, status: int, content_type: str, body: bytes) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args: object) -> None:
            # Health checks every few seconds would drown the real logs.
            return

    server = ThreadingHTTPServer((host, port), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server
