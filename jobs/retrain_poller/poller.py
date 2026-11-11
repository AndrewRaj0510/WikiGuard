"""Pick up what the API asked for. Render holds no cluster credentials, so this is the only way in.

The API adds a row to retrain_requests, and this job, inside the cluster, polls for it. A
compromised API can then only ask: it can't reach the cluster, and this code decides what a request
may do.

- A row is claimed with FOR UPDATE SKIP LOCKED, so two runs of this job can't take the same one.
- At most one retrain runs per 24 hours, whoever asks (section 5 of the plan).
- A run that dies without finishing is failed after a time limit, so it can't block retraining.

Every kind runs TRAINER_COMMAND with the kind and the request's id added, for example
`python -m trainer.lifecycle retrain --request 12 --run 4`. The trainer reads the rest from the row,
so nothing a request says can become part of a command. The monitor asks the same way the API does,
so this is the only thing that starts the trainer or moves an alias (DEC-071).

This job also asks for one thing itself: a shadow_check once a candidate's shadow period is over.
"""

import logging
import shlex
import subprocess
import sys
from collections.abc import Callable
from typing import Any

import psycopg
from psycopg.types.json import Jsonb
from wikiguard_core.config import (
    CommonSettings,
    ConfigError,
    DatabaseSettings,
    RetrainSettings,
    load_settings,
)
from wikiguard_core.identity import RedactingFilter

logger = logging.getLogger("retrain_poller")

RETRAIN_GAP = "24 hours"
STUCK_AFTER = "12 hours"
TRAINER_TIMEOUT_SECONDS = 12 * 60 * 60
Launcher = Callable[[dict[str, Any]], None]


class Settings(CommonSettings, DatabaseSettings, RetrainSettings):
    pass


class NotBuiltYet(RuntimeError):
    pass


KINDS = ("retrain", "promote", "rollback", "shadow_check")


def command_arguments(command: str, request: dict[str, Any]) -> list[str]:
    """Only the kind and numbers from the row, never its text."""
    kind = str(request["kind"])
    if kind not in KINDS:
        raise NotBuiltYet(f"no handler for {kind}")
    args = [*shlex.split(command), kind.replace("_", "-"), "--request", str(int(request["id"]))]
    if request.get("run_id") is not None:
        args += ["--run", str(int(request["run_id"]))]
    return args


def command_launcher(command: str | None) -> Launcher:
    """Runs the trainer as a command. Phase 7 swaps this for creating a Kubernetes Job."""

    def launch(request: dict[str, Any]) -> None:
        if not command:
            raise NotBuiltYet("TRAINER_COMMAND isn't set")
        # A list and no shell, so a request's text can never become part of a command.
        subprocess.run(  # noqa: S603 - the command comes from the operator's settings
            command_arguments(command, request), check=True, timeout=TRAINER_TIMEOUT_SECONDS
        )

    return launch


def not_built_launcher(request: dict[str, Any]) -> None:
    raise NotBuiltYet(f"no launcher for {request['kind']}: TRAINER_COMMAND isn't set")


def launchers_for(command: str | None) -> dict[str, Launcher]:
    return dict.fromkeys(KINDS, command_launcher(command))


def request_shadow_checks(connection: psycopg.Connection[Any], shadow_hours: int) -> int:
    """Ask for a check of every candidate whose shadow period is over, once each."""
    count = connection.execute(
        """
        INSERT INTO retrain_requests (kind, reason, requested_by, details)
        SELECT 'shadow_check', 'shadow period over', 'retrain_poller',
               jsonb_build_object('run_id', r.id)
        FROM retrain_runs r
        WHERE r.status = 'shadow'
          AND (r.details->>'shadow_started_at')::timestamptz
              <= now() - make_interval(hours => %s)
          AND NOT EXISTS (
              SELECT 1 FROM retrain_requests q
              WHERE q.kind = 'shadow_check' AND q.status IN ('pending', 'running')
                AND (q.details->>'run_id')::bigint = r.id
          )
        """,
        (shadow_hours,),
    ).rowcount
    connection.commit()
    return count


def fail_stuck(connection: psycopg.Connection[Any]) -> int:
    count = connection.execute(
        """
        UPDATE retrain_requests
        SET status = 'failed', finished_at = now(), details = details || %s
        WHERE status = 'running' AND picked_up_at < now() - %s::interval
        """,
        (Jsonb({"error": "no result before the time limit"}), STUCK_AFTER),
    ).rowcount
    connection.commit()
    return count


def claim_next(connection: psycopg.Connection[Any]) -> dict[str, Any] | None:
    row = connection.execute(
        """
        UPDATE retrain_requests SET status = 'running', picked_up_at = now()
        WHERE id = (
            SELECT id FROM retrain_requests WHERE status = 'pending'
            ORDER BY requested_at LIMIT 1 FOR UPDATE SKIP LOCKED
        )
        RETURNING id, kind, reason, requested_by, details
        """
    ).fetchone()
    connection.commit()
    if row is None:
        return None
    return dict(zip(("id", "kind", "reason", "requested_by", "details"), row, strict=True))


def retrained_recently(connection: psycopg.Connection[Any], own_id: int) -> bool:
    row = connection.execute(
        """
        SELECT 1 FROM retrain_requests
        WHERE kind = 'retrain' AND id <> %s
          AND (status = 'running' OR (status = 'done' AND finished_at > now() - %s::interval))
        LIMIT 1
        """,
        (own_id, RETRAIN_GAP),
    ).fetchone()
    connection.commit()
    return row is not None


def finish(
    connection: psycopg.Connection[Any], request_id: int, status: str, details: dict[str, Any]
) -> None:
    connection.execute(
        "UPDATE retrain_requests SET status = %s, finished_at = now(), details = details || %s "
        "WHERE id = %s",
        (status, Jsonb(details), request_id),
    )
    connection.execute(
        "INSERT INTO audit_log (actor, action, target, details) "
        "VALUES ('retrain_poller', %s, %s, %s)",
        (f"request.{status}", str(request_id), Jsonb(details)),
    )
    connection.commit()


def run_once(
    connection: psycopg.Connection[Any],
    launchers: dict[str, Launcher],
) -> dict[str, Any] | None:
    """Handle one request. Returns it with its outcome, or None if nothing was waiting."""
    fail_stuck(connection)
    request = claim_next(connection)
    if request is None:
        return None
    request_id = request["id"]
    kind = request["kind"]
    if kind == "retrain" and retrained_recently(connection, request_id):
        finish(connection, request_id, "failed", {"error": f"one retrain per {RETRAIN_GAP}"})
        return request | {"status": "failed"}
    run_id = None
    if kind == "retrain":
        asked = request["details"] or {}
        # The monitor names its trigger. A person asking is a manual retrain.
        trigger = asked.get("trigger") if request["requested_by"] == "monitor" else "manual"
        row = connection.execute(
            "INSERT INTO retrain_runs (trigger, status, details) VALUES (%s, 'running', %s) "
            "RETURNING id",
            (
                trigger or "manual",
                Jsonb(
                    {
                        "request_id": request_id,
                        "reason": request["reason"],
                        "reasons": asked.get("reasons", []),
                    }
                ),
            ),
        ).fetchone()
        connection.commit()
        run_id = row[0] if row else None
    elif kind == "shadow_check":
        run_id = (request["details"] or {}).get("run_id")
    try:
        launchers.get(kind, not_built_launcher)(request | {"run_id": run_id})
    except Exception as error:  # noqa: BLE001 - recorded on the request, for the Health page
        outcome, details = "failed", {"error": type(error).__name__, "message": str(error)[:200]}
        logger.error("Request %s failed: %s", request_id, type(error).__name__)
    else:
        outcome, details = "done", {}
    if run_id is not None and kind == "retrain":
        # The trainer sets its own outcome (shadow, promoted, rejected...). Only a run it never
        # got to, or one it crashed in, is closed here.
        connection.execute(
            "UPDATE retrain_runs SET status = %s, finished_at = now() "
            "WHERE id = %s AND status = 'running'",
            (outcome, run_id),
        )
    finish(connection, request_id, outcome, details | ({"run_id": run_id} if run_id else {}))
    return request | {"status": outcome}


def main() -> int:
    handler = logging.StreamHandler()
    handler.addFilter(RedactingFilter())
    logging.basicConfig(level=logging.INFO, handlers=[handler])
    try:
        settings = load_settings(Settings)
    except ConfigError as error:
        logger.error("%s", error)
        return 2
    launchers = launchers_for(settings.trainer_command)
    with psycopg.connect(
        settings.database_url.get_secret_value(), prepare_threshold=None, connect_timeout=10
    ) as connection:
        if request_shadow_checks(connection, settings.shadow_hours):
            logger.info("A shadow period ended, so its check was requested")
        while (result := run_once(connection, launchers)) is not None:
            logger.info("Request %s: %s", result["id"], result["status"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
