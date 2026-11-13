"""The API's shared parts: settings, RFC 9457 errors, the database, rate limits, and JSON logs."""

import json
import logging
import threading
import time
from collections import OrderedDict
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any

import psycopg
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool, PoolClosed, PoolTimeout
from wikiguard_core.config import (
    AlertSettings,
    ApiSettings,
    CommonSettings,
    DatabaseSettings,
    FeatureFlagSettings,
    GateSettings,
    ServingSettings,
)

PROBLEM_TYPE = "application/problem+json"
RATE_TABLE_LIMIT = 50_000


class Settings(
    CommonSettings,
    DatabaseSettings,
    ServingSettings,
    ApiSettings,
    FeatureFlagSettings,
    GateSettings,
    AlertSettings,
):
    pass


class Problem(Exception):
    """An error for the client, in one format. Details are written here, never copied from input."""

    def __init__(
        self,
        status: int,
        code: str,
        title: str,
        detail: str | None = None,
        retry_after: int | None = None,
        extra: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(title)
        self.status = status
        self.code = code
        self.title = title
        self.detail = detail
        self.retry_after = retry_after
        self.extra = extra or {}


def problem_body(
    status: int,
    code: str,
    title: str,
    request_id: str | None,
    detail: str | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "type": f"/problems/{code}",
        "title": title,
        "status": status,
        "code": code,
        "request_id": request_id,
    }
    if detail:
        body["detail"] = detail
    return body | (extra or {})


class DatabaseUnavailable(Problem):
    def __init__(self) -> None:
        super().__init__(
            503,
            "database_unavailable",
            "The database is unavailable",
            "Try again shortly.",
            retry_after=5,
        )


class Database:
    """A small pool. Each transaction sets its own limit, because Supabase's transaction pooler
    shares connections, so a session-level setting could leak onto someone else's queries."""

    def __init__(
        self,
        url: str,
        statement_timeout_ms: int = 3000,
        max_size: int = 5,
        on_statement: Callable[[str], None] | None = None,
    ) -> None:
        self.statement_timeout_ms = statement_timeout_ms
        self._on_statement = on_statement
        self.pool = ConnectionPool(
            url,
            min_size=0,
            max_size=max_size,
            open=False,
            timeout=3.0,
            max_idle=60.0,
            reconnect_timeout=10.0,
            # No prepared statements, since the pooler can't hold them across connections.
            kwargs={"prepare_threshold": None, "connect_timeout": 5, "row_factory": dict_row},
        )

    def open(self) -> None:
        self.pool.open(wait=False)

    def close(self) -> None:
        self.pool.close()

    @contextmanager
    def tx(self) -> Iterator["Tx"]:
        """One transaction. Commits at the end, and rolls back if anything raises."""
        try:
            with self.pool.connection() as connection:
                connection.execute(
                    "SELECT set_config('statement_timeout', %s, true)",
                    (str(self.statement_timeout_ms),),
                )
                yield Tx(connection, self._on_statement)
        except (psycopg.OperationalError, psycopg.InterfaceError, PoolTimeout, PoolClosed) as error:
            raise DatabaseUnavailable from error
        except psycopg.errors.QueryCanceled as error:
            raise DatabaseUnavailable from error

    def check(self) -> bool:
        try:
            with self.tx() as tx:
                tx.one("SELECT 1 AS up")
        except DatabaseUnavailable:
            return False
        return True


class Tx:
    def __init__(
        self, connection: psycopg.Connection[Any], on_statement: Callable[[str], None] | None
    ) -> None:
        self.connection = connection
        self._on_statement = on_statement

    def _run(self, query: str, params: Any = None) -> psycopg.Cursor[Any]:
        if self._on_statement is not None:
            self._on_statement(query)
        return self.connection.execute(query, params)

    def one(self, query: str, params: Any = None) -> dict[str, Any] | None:
        row: dict[str, Any] | None = self._run(query, params).fetchone()
        return row

    def all(self, query: str, params: Any = None) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = self._run(query, params).fetchall()
        return rows

    def run(self, query: str, params: Any = None) -> int:
        return self._run(query, params).rowcount


class RateLimiter:
    """A fixed window per key. In memory, since the API runs as one process (DEC-016).

    It resets when Render sleeps, which costs an attacker a cold start each time. Limits that must
    survive a restart live in Postgres: the login lockout and the LLM call counter.
    """

    def __init__(self, clock: Callable[[], float] = time.monotonic) -> None:
        self._clock = clock
        self._windows: OrderedDict[str, tuple[float, int]] = OrderedDict()
        self._lock = threading.Lock()

    def hit(self, key: str, limit: int, window_seconds: int) -> int | None:
        """Count one request. Returns seconds to wait if it's over the limit, otherwise None."""
        now = self._clock()
        with self._lock:
            started, count = self._windows.get(key, (now, 0))
            if now - started >= window_seconds:
                started, count = now, 0
            count += 1
            self._windows[key] = (started, count)
            self._windows.move_to_end(key)
            # Bounded, so requests from many IPs can't grow this without limit.
            while len(self._windows) > RATE_TABLE_LIMIT:
                self._windows.popitem(last=False)
            if count > limit:
                return max(int(window_seconds - (now - started)) + 1, 1)
        return None

    def require(self, key: str, limit: int, window_seconds: int) -> None:
        wait = self.hit(key, limit, window_seconds)
        if wait is not None:
            raise Problem(
                429,
                "rate_limited",
                "Too many requests",
                "Wait a moment before trying again.",
                retry_after=wait,
            )


class JsonFormatter(logging.Formatter):
    """One JSON object per line. Only named fields are added, so a diff can't slip in."""

    FIELDS = ("request_id", "method", "route", "status", "duration_ms", "rev_id", "role")

    def format(self, record: logging.LogRecord) -> str:
        body: dict[str, Any] = {
            "at": self.formatTime(record),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for name in self.FIELDS:
            value = getattr(record, name, None)
            if value is not None:
                body[name] = value
        if record.exc_info and record.exc_info[0] is not None:
            # The type only: a traceback's values could include request data.
            body["error"] = record.exc_info[0].__name__
        return json.dumps(body)
