"""Alerts for the jobs: always to the log, and to Discord when a webhook is set (DEC-023).

An alert must never stop the job that raised it, so sending never raises. Callers pass names and
numbers only: diffs, titles, and usernames are attacker-controlled and never go into an alert.
"""

import logging
from dataclasses import dataclass, field
from typing import Literal

import httpx

logger = logging.getLogger(__name__)

Severity = Literal["info", "warning", "critical"]
# Discord refuses messages over 2,000 characters.
MAX_CONTENT = 1_900
TIMEOUT_SECONDS = 10.0


@dataclass(frozen=True)
class Alert:
    source: str
    title: str
    detail: str
    severity: Severity


@dataclass
class Alerter:
    source: str
    webhook_url: str | None = None
    transport: httpx.BaseTransport | None = None
    sent: list[Alert] = field(default_factory=list)

    def send(self, title: str, detail: str, severity: Severity = "warning") -> bool:
        """Returns whether Discord took it. The log line and the record happen either way."""
        alert = Alert(self.source, title, detail, severity)
        self.sent.append(alert)
        log = logger.error if severity == "critical" else logger.warning
        log("ALERT [%s] %s: %s", severity, title, detail)
        if not self.webhook_url:
            return False
        content = f"**[{severity}] {self.source}: {title}**\n{detail}"[:MAX_CONTENT]
        try:
            with httpx.Client(timeout=TIMEOUT_SECONDS, transport=self.transport) as client:
                # No mentions are parsed, so text in an alert can't ping anyone.
                response = client.post(
                    self.webhook_url,
                    json={"content": content, "allowed_mentions": {"parse": []}},
                )
                response.raise_for_status()
        except httpx.HTTPError as error:
            logger.warning("Alert not delivered: %s", type(error).__name__)
            return False
        return True


def ping(url: str | None, transport: httpx.BaseTransport | None = None) -> bool:
    """A Healthchecks.io heartbeat: an alert fires there when the pings stop (DEC-023)."""
    if not url:
        return False
    try:
        with httpx.Client(timeout=TIMEOUT_SECONDS, transport=transport) as client:
            client.get(url).raise_for_status()
    except httpx.HTTPError as error:
        logger.warning("Heartbeat not delivered: %s", type(error).__name__)
        return False
    return True
