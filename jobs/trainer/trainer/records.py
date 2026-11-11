"""What the trainer writes to Postgres: runs, the audit log, and the model list the API shows.

Everything goes through Recorder, so the pipeline's tests run with MemoryRecorder and the database
tests check DbRecorder against the real trainer role's grants.
"""

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Protocol

from psycopg.types.json import Jsonb

ACTOR = "trainer"


@dataclass(frozen=True)
class Listing:
    """One model_versions row. Empty metrics keep whatever the row already has."""

    model_name: str
    model_version: str
    feature_version: int
    snapshot_hash: str
    artifact_sha256: str
    metrics: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ShadowRow:
    score: float | None
    error: str | None
    latency_ms: float | None
    served_score: float | None


class Recorder(Protocol):
    def run(self, run_id: int) -> dict[str, Any]: ...

    def start_run(self, trigger: str, status: str, details: dict[str, Any]) -> int: ...

    def update_run(
        self,
        run_id: int,
        status: str | None = None,
        details: dict[str, Any] | None = None,
        finished: bool = False,
    ) -> None: ...

    def request(self, request_id: int) -> dict[str, Any]: ...

    def audit(self, action: str, target: str | None, details: dict[str, Any]) -> None: ...

    def sync_models(self, aliases: dict[str, str], listings: list[Listing]) -> None: ...

    def model_metrics(self) -> dict[str, dict[str, Any]]:
        """metrics by version, from model_versions."""
        ...

    def shadow_rows(self, version: str, since: datetime) -> list[ShadowRow]: ...

    def promotions(self) -> list[dict[str, Any]]:
        """Runs that moved the champion, newest first."""
        ...


class DbRecorder:
    def __init__(self, connection: Any) -> None:
        self.connection = connection

    def _one(self, query: str, params: tuple[Any, ...]) -> dict[str, Any] | None:
        cursor = self.connection.execute(query, params)
        row = cursor.fetchone()
        names = [column.name for column in cursor.description or []]
        self.connection.commit()
        return dict(zip(names, row, strict=True)) if row else None

    def run(self, run_id: int) -> dict[str, Any]:
        row = self._one(
            "SELECT id, trigger, status, started_at, finished_at, details "
            "FROM retrain_runs WHERE id = %s",
            (run_id,),
        )
        if row is None:
            raise LookupError(f"no retrain run {run_id}")
        return row

    def start_run(self, trigger: str, status: str, details: dict[str, Any]) -> int:
        row = self._one(
            "INSERT INTO retrain_runs (trigger, status, details) VALUES (%s, %s, %s) RETURNING id",
            (trigger, status, Jsonb(details)),
        )
        assert row is not None  # noqa: S101 - RETURNING on an INSERT
        return int(row["id"])

    def update_run(
        self,
        run_id: int,
        status: str | None = None,
        details: dict[str, Any] | None = None,
        finished: bool = False,
    ) -> None:
        self.connection.execute(
            """
            UPDATE retrain_runs SET
                status = coalesce(%s, status),
                details = details || %s,
                finished_at = CASE WHEN %s THEN now() ELSE finished_at END
            WHERE id = %s
            """,
            (status, Jsonb(details or {}), finished, run_id),
        )
        self.connection.commit()

    def request(self, request_id: int) -> dict[str, Any]:
        row = self._one(
            "SELECT id, kind, reason, requested_by, details FROM retrain_requests WHERE id = %s",
            (request_id,),
        )
        if row is None:
            raise LookupError(f"no request {request_id}")
        return row

    def audit(self, action: str, target: str | None, details: dict[str, Any]) -> None:
        self.connection.execute(
            "INSERT INTO audit_log (actor, action, target, details) VALUES (%s, %s, %s, %s)",
            (ACTOR, action, target, Jsonb(details)),
        )
        self.connection.commit()

    def sync_models(self, aliases: dict[str, str], listings: list[Listing]) -> None:
        """The API's copy of the registry. Every alias moves at once, in one transaction."""
        by_version = {listing.model_version: listing for listing in listings}
        with self.connection.transaction():
            self.connection.execute(
                "UPDATE model_versions SET alias = NULL WHERE alias IS NOT NULL"
            )
            for alias, version in aliases.items():
                listing = by_version[version]
                self.connection.execute(
                    """
                    INSERT INTO model_versions (model_name, model_version, alias, feature_version,
                        snapshot_hash, artifact_sha256, metrics)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (model_name, model_version) DO UPDATE SET
                        alias = EXCLUDED.alias,
                        metrics = CASE WHEN EXCLUDED.metrics = '{}'::jsonb
                                       THEN model_versions.metrics ELSE EXCLUDED.metrics END
                    """,
                    (
                        listing.model_name,
                        listing.model_version,
                        alias,
                        listing.feature_version,
                        listing.snapshot_hash,
                        listing.artifact_sha256,
                        Jsonb(listing.metrics),
                    ),
                )
        self.connection.commit()

    def model_metrics(self) -> dict[str, dict[str, Any]]:
        cursor = self.connection.execute("SELECT model_version, metrics FROM model_versions")
        rows = {str(version): metrics for version, metrics in cursor.fetchall()}
        self.connection.commit()
        return rows

    def shadow_rows(self, version: str, since: datetime) -> list[ShadowRow]:
        cursor = self.connection.execute(
            "SELECT score, error, latency_ms, served_score FROM shadow_scores "
            "WHERE model_version = %s AND scored_at >= %s",
            (version, since),
        )
        rows = [ShadowRow(*row) for row in cursor.fetchall()]
        self.connection.commit()
        return rows

    def promotions(self) -> list[dict[str, Any]]:
        cursor = self.connection.execute(
            "SELECT id, trigger, status, started_at, details FROM retrain_runs "
            "WHERE status IN ('promoted', 'rolled_back') AND details ? 'aliases_before' "
            "ORDER BY (details->>'promoted_at') DESC NULLS LAST, id DESC"
        )
        names = [column.name for column in cursor.description or []]
        rows = [dict(zip(names, row, strict=True)) for row in cursor.fetchall()]
        self.connection.commit()
        return rows


@dataclass
class MemoryRecorder:
    """The same contract in memory, for the pipeline's tests."""

    runs: dict[int, dict[str, Any]] = field(default_factory=dict)
    requests: dict[int, dict[str, Any]] = field(default_factory=dict)
    audits: list[tuple[str, str | None, dict[str, Any]]] = field(default_factory=list)
    listed: dict[str, dict[str, Any]] = field(default_factory=dict)
    shadow: dict[str, list[ShadowRow]] = field(default_factory=dict)

    def run(self, run_id: int) -> dict[str, Any]:
        return self.runs[run_id]

    def start_run(self, trigger: str, status: str, details: dict[str, Any]) -> int:
        run_id = len(self.runs) + 1
        self.runs[run_id] = {
            "id": run_id,
            "trigger": trigger,
            "status": status,
            "started_at": datetime.now(UTC),
            "finished_at": None,
            "details": dict(details),
        }
        return run_id

    def update_run(
        self,
        run_id: int,
        status: str | None = None,
        details: dict[str, Any] | None = None,
        finished: bool = False,
    ) -> None:
        run = self.runs[run_id]
        if status is not None:
            run["status"] = status
        run["details"] = run["details"] | (details or {})
        if finished:
            run["finished_at"] = datetime.now(UTC)

    def request(self, request_id: int) -> dict[str, Any]:
        return self.requests[request_id]

    def audit(self, action: str, target: str | None, details: dict[str, Any]) -> None:
        self.audits.append((action, target, details))

    def sync_models(self, aliases: dict[str, str], listings: list[Listing]) -> None:
        for row in self.listed.values():
            row["alias"] = None
        for listing in listings:
            row = self.listed.setdefault(
                listing.model_version, {"model_name": listing.model_name, "metrics": {}}
            )
            if listing.metrics:
                row["metrics"] = listing.metrics
        for alias, version in aliases.items():
            self.listed[version]["alias"] = alias

    def model_metrics(self) -> dict[str, dict[str, Any]]:
        return {version: row["metrics"] for version, row in self.listed.items()}

    def shadow_rows(self, version: str, since: datetime) -> list[ShadowRow]:
        return list(self.shadow.get(version, []))

    def promotions(self) -> list[dict[str, Any]]:
        found = [
            run
            for run in self.runs.values()
            if run["status"] in ("promoted", "rolled_back") and "aliases_before" in run["details"]
        ]
        return sorted(
            found, key=lambda run: (run["details"].get("promoted_at", ""), run["id"]), reverse=True
        )
