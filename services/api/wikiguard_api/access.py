"""Sign-in, server-side sessions, and the single seat (section 16 of the plan).

Every rule that matters lives in a SQL statement, so it holds across restarts and at the same
instant in two requests:

- The seat is one row. Claiming it is INSERT ... ON CONFLICT DO UPDATE ... WHERE, so of two
  simultaneous sign-ins, the second waits for the first and then finds the seat taken.
- Only an HMAC of the cookie is stored, so a leaked table can't be replayed as a session.
- Times come from the database's now(), so there's one clock.
- A heartbeat keeps the lease, but only real activity resets a guest's idle timer. A forgotten tab
  therefore can't hold the seat. The total limit also covers a client lying about activity.
"""

import hashlib
import hmac
import secrets
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from argon2 import PasswordHasher
from argon2.exceptions import InvalidHashError, VerificationError, VerifyMismatchError

from wikiguard_api.foundations import Database, DatabaseUnavailable, Problem, Settings, Tx

GUEST = "guest"
USER = "user"
USER_SESSION_HOURS = 12
IP_FAILURE_LIMIT = 5
IP_LOCK_MINUTES = 15
# Across all IPs: a brake on distributed guessing. Short, because it also locks out the owner.
GLOBAL_FAILURE_LIMIT = 30
GLOBAL_LOCK_MINUTES = 5
THROTTLE_WINDOW = "1 hour"
MAX_PASSWORD_LENGTH = 256


@dataclass(frozen=True)
class Principal:
    session_id: str
    role: str
    csrf_token: str
    created_at: datetime
    expires_at: datetime
    fresh_explains: int
    # None when the gate is off.
    tab_id: str | None = None


def session_id_for(token: str, key: str) -> str:
    return hmac.new(key.encode(), token.encode(), hashlib.sha256).hexdigest()


def client_hash(ip: str, key: str) -> str:
    return hmac.new(key.encode(), f"client:{ip}".encode(), hashlib.sha256).hexdigest()


class Passwords:
    """Argon2id takes real CPU and memory on purpose, so only one check runs at a time. A flood of
    guesses gets a quick refusal, and can't exhaust a 512 MB service."""

    def __init__(self, stored_hash: str) -> None:
        self._stored = stored_hash
        self._hasher = PasswordHasher()
        self._slot = threading.BoundedSemaphore(1)

    def verify(self, password: str) -> bool:
        if not self._slot.acquire(blocking=False):
            raise Problem(503, "login_busy", "Sign-in is busy", "Try again.", retry_after=2)
        try:
            return self._hasher.verify(self._stored, password)
        except (VerifyMismatchError, VerificationError, InvalidHashError):
            return False
        finally:
            self._slot.release()


def audit(
    tx: Tx,
    actor: str,
    action: str,
    target: str | None = None,
    request_id: str | None = None,
    details: dict[str, Any] | None = None,
) -> None:
    from psycopg.types.json import Jsonb

    tx.run(
        "INSERT INTO audit_log (actor, action, target, request_id, details) "
        "VALUES (%s, %s, %s, %s, %s)",
        (actor, action, target, request_id, Jsonb(details or {})),
    )


class Access:
    def __init__(self, settings: Settings, db: Database) -> None:
        self.settings = settings
        self.db = db
        self.passwords = Passwords(settings.user_password_hash.get_secret_value())
        self._key = settings.session_secret_key.get_secret_value()
        self._lease = f"{settings.gate_lease_seconds} seconds"
        self._idle = f"{settings.guest_idle_minutes} minutes"
        self._max = f"{settings.guest_max_minutes} minutes"
        # What we knew at the last good check, so a database outage doesn't sign anyone out early.
        self._recent: dict[tuple[str, str | None], tuple[Principal, float]] = {}
        self._recent_lock = threading.Lock()

    def hash_ip(self, ip: str) -> str:
        return client_hash(ip, self._key)

    # --- Housekeeping ---

    def _sweep(self, tx: Tx) -> None:
        """End guest sessions past their limits, and any session past its expiry, and free seats."""
        tx.run(
            """
            UPDATE sessions SET ended_at = now(), ended_reason = 'expired'
            WHERE ended_at IS NULL AND (
                expires_at <= now()
                OR (role = 'guest' AND (
                    created_at <= now() - %s::interval OR last_activity_at <= now() - %s::interval
                ))
            )
            """,
            (self._max, self._idle),
        )
        tx.run(
            "DELETE FROM active_session a USING sessions s "
            "WHERE a.session_id = s.id AND s.ended_at IS NOT NULL"
        )

    def _holder(self, tx: Tx) -> dict[str, Any] | None:
        """Who holds a live lease. An expired lease means the seat is free, after a crash say."""
        return tx.one(
            """
            SELECT a.session_id, a.role, a.tab_id, s.created_at, s.last_activity_at,
                GREATEST(0, EXTRACT(EPOCH FROM LEAST(
                    s.created_at + %s::interval, s.last_activity_at + %s::interval
                ) - now()))::int AS guest_seconds_left
            FROM active_session a JOIN sessions s ON s.id = a.session_id
            WHERE a.lease_expires_at > now()
            """,
            (self._max, self._idle),
        )

    def _claim(self, tx: Tx, session_id: str, role: str, tab_id: str, force: bool = False) -> bool:
        """True if this session now holds the seat. force is for `user` only: callers check that."""
        row = tx.one(
            """
            INSERT INTO active_session (seat, session_id, role, tab_id, lease_expires_at)
            VALUES (1, %(sid)s, %(role)s, %(tab)s, now() + %(lease)s::interval)
            ON CONFLICT (seat) DO UPDATE SET
                session_id = EXCLUDED.session_id,
                role = EXCLUDED.role,
                tab_id = EXCLUDED.tab_id,
                lease_expires_at = EXCLUDED.lease_expires_at,
                started_at = now(),
                last_activity_at = now()
            WHERE active_session.lease_expires_at <= now() OR %(force)s
            RETURNING session_id
            """,
            {"sid": session_id, "role": role, "tab": tab_id, "lease": self._lease, "force": force},
        )
        return row is not None

    def _end_session(self, tx: Tx, session_id: str, reason: str) -> None:
        tx.run(
            "UPDATE sessions SET ended_at = now(), ended_reason = %s "
            "WHERE id = %s AND ended_at IS NULL",
            (reason, session_id),
        )
        tx.run("DELETE FROM active_session WHERE session_id = %s", (session_id,))

    def _new_session(self, tx: Tx, role: str, ip_hash: str) -> tuple[str, str]:
        token = secrets.token_urlsafe(32)
        csrf = secrets.token_urlsafe(32)
        lifetime = self._max if role == GUEST else f"{USER_SESSION_HOURS} hours"
        tx.run(
            "INSERT INTO sessions (id, role, csrf_token, client_hash, expires_at) "
            "VALUES (%s, %s, %s, %s, now() + %s::interval)",
            (session_id_for(token, self._key), role, csrf, ip_hash, lifetime),
        )
        return token, csrf

    @staticmethod
    def _seat_taken(holder: dict[str, Any]) -> Problem:
        left = holder["guest_seconds_left"] if holder["role"] == GUEST else None
        return Problem(
            409,
            "seat_taken",
            "WikiGuard is in use",
            "Only one person can use the app at a time.",
            retry_after=left,
            extra={"retry_after_seconds": left},
        )

    # --- Signing in ---

    def sign_in_guest(self, ip: str, tab_id: str, request_id: str | None) -> tuple[str, str]:
        ip_hash = self.hash_ip(ip)
        with self.db.tx() as tx:
            self._sweep(tx)
            wait = tx.one(
                """
                SELECT CEIL(EXTRACT(EPOCH FROM ended_at + %s::interval - now()))::int AS seconds
                FROM sessions
                WHERE role = 'guest' AND client_hash = %s AND ended_at IS NOT NULL
                  AND ended_at > now() - %s::interval
                ORDER BY ended_at DESC LIMIT 1
                """,
                (
                    f"{self.settings.guest_signin_wait_seconds} seconds",
                    ip_hash,
                    f"{self.settings.guest_signin_wait_seconds} seconds",
                ),
            )
            if wait is not None:
                raise Problem(
                    429,
                    "guest_cooldown",
                    "Please wait before signing in again",
                    retry_after=max(int(wait["seconds"]), 1),
                )
            token, csrf = self._new_session(tx, GUEST, ip_hash)
            if self.settings.gate_enabled:
                session_id = session_id_for(token, self._key)
                if not self._claim(tx, session_id, GUEST, tab_id):
                    holder = self._holder(tx)
                    # Raising rolls the new session back, so a refused visitor leaves no row.
                    raise (
                        self._seat_taken(holder)
                        if holder
                        else Problem(409, "seat_taken", "WikiGuard is in use", retry_after=5)
                    )
            audit(tx, GUEST, "auth.sign_in", request_id=request_id)
        return token, csrf

    def _locked(self, tx: Tx, scope: str) -> int | None:
        row = tx.one(
            "SELECT CEIL(EXTRACT(EPOCH FROM locked_until - now()))::int AS seconds "
            "FROM login_throttle WHERE scope = %s AND locked_until > now()",
            (scope,),
        )
        return int(row["seconds"]) if row else None

    def _record_failure(self, tx: Tx, scope: str, limit: int, lock_minutes: int) -> None:
        tx.run(
            """
            INSERT INTO login_throttle (scope, failures) VALUES (%(scope)s, 1)
            ON CONFLICT (scope) DO UPDATE SET
                failures = CASE WHEN login_throttle.window_start < now() - %(window)s::interval
                    THEN 1 ELSE login_throttle.failures + 1 END,
                window_start = CASE WHEN login_throttle.window_start < now() - %(window)s::interval
                    THEN now() ELSE login_throttle.window_start END,
                locked_until = CASE WHEN login_throttle.failures + 1 >= %(limit)s
                    THEN now() + %(lock)s::interval ELSE login_throttle.locked_until END
            """,
            {
                "scope": scope,
                "window": THROTTLE_WINDOW,
                "limit": limit,
                "lock": f"{lock_minutes} minutes",
            },
        )

    def sign_in_user(
        self, password: str, ip: str, tab_id: str, take_over: bool, request_id: str | None
    ) -> tuple[str, str]:
        ip_hash = self.hash_ip(ip)
        ip_scope = f"ip:{ip_hash}"
        with self.db.tx() as tx:
            locked = self._locked(tx, ip_scope) or self._locked(tx, "global")
        # Refused before any hashing, so a locked-out attacker costs no CPU.
        if locked is not None:
            raise Problem(
                429, "login_locked", "Too many failed sign-ins", retry_after=max(locked, 1)
            )
        # Outside any transaction: this takes a while, and shouldn't hold a connection.
        if not self.passwords.verify(password[:MAX_PASSWORD_LENGTH]):
            with self.db.tx() as tx:
                self._record_failure(tx, ip_scope, IP_FAILURE_LIMIT, IP_LOCK_MINUTES)
                self._record_failure(tx, "global", GLOBAL_FAILURE_LIMIT, GLOBAL_LOCK_MINUTES)
                audit(tx, "anonymous", "auth.sign_in_failed", request_id=request_id)
            raise Problem(401, "bad_credentials", "Wrong password")
        with self.db.tx() as tx:
            tx.run("DELETE FROM login_throttle WHERE scope = %s", (ip_scope,))
            self._sweep(tx)
            holder = self._holder(tx) if self.settings.gate_enabled else None
            if holder is not None and not take_over:
                raise Problem(
                    409,
                    "confirm_take_over",
                    "WikiGuard is in use",
                    "Confirm to end the other session and take the seat.",
                    extra={"holder_role": holder["role"]},
                )
            token, csrf = self._new_session(tx, USER, ip_hash)
            session_id = session_id_for(token, self._key)
            if holder is not None:
                self._end_session(tx, holder["session_id"], "taken_over")
                audit(tx, USER, "gate.take_over", holder["role"], request_id)
            if self.settings.gate_enabled:
                self._claim(tx, session_id, USER, tab_id, force=True)
            audit(tx, USER, "auth.sign_in", request_id=request_id)
        return token, csrf

    def sign_out(self, principal: Principal, request_id: str | None) -> None:
        with self.db.tx() as tx:
            self._end_session(tx, principal.session_id, "signed_out")
            audit(tx, principal.role, "auth.sign_out", request_id=request_id)
        self._forget(principal.session_id)

    # --- Each request ---

    def _forget(self, session_id: str) -> None:
        with self._recent_lock:
            for key in [key for key in self._recent if key[0] == session_id]:
                del self._recent[key]

    def _remember(self, principal: Principal, tab_id: str | None, lease_seconds: float) -> None:
        with self._recent_lock:
            # One active session at a time, so this stays tiny.
            if len(self._recent) > 16:
                self._recent.clear()
            key = (principal.session_id, tab_id)
            self._recent[key] = (principal, time.monotonic() + lease_seconds)

    def session(self, token: str | None) -> Principal:
        """The signed-in session, with no seat check. Used by the gate's own endpoints."""
        principal, _ = self._load(token)
        return principal

    def _load(self, token: str | None) -> tuple[Principal, dict[str, Any]]:
        if not token:
            raise Problem(401, "not_signed_in", "Sign in first")
        session_id = session_id_for(token, self._key)
        with self.db.tx() as tx:
            row = tx.one(
                """
                SELECT s.id, s.role, s.csrf_token, s.created_at, s.expires_at, s.fresh_explains,
                    s.ended_reason,
                    (s.ended_at IS NOT NULL OR s.expires_at <= now()
                     OR (s.role = 'guest' AND (
                        s.created_at <= now() - %s::interval
                        OR s.last_activity_at <= now() - %s::interval))) AS is_over,
                    (s.ended_at IS NOT NULL) AS already_ended,
                    a.tab_id AS seat_tab,
                    (a.lease_expires_at > now()) AS lease_live,
                    GREATEST(0, EXTRACT(EPOCH FROM a.lease_expires_at - now())) AS lease_left
                FROM sessions s LEFT JOIN active_session a ON a.session_id = s.id
                WHERE s.id = %s
                """,
                (self._max, self._idle, session_id),
            )
            if row is None:
                raise Problem(401, "not_signed_in", "Sign in first")
            if row["is_over"]:
                if not row["already_ended"]:
                    self._end_session(tx, session_id, "expired")
                reason = row["ended_reason"] or "expired"
                # Raised after the block's commit, so ending the session isn't rolled back.
                expired = Problem(
                    401, "session_ended", "Your session has ended", extra={"reason": reason}
                )
            else:
                expired = None
        if expired is not None:
            self._forget(session_id)
            raise expired
        return (
            Principal(
                row["id"],
                row["role"],
                row["csrf_token"],
                row["created_at"],
                row["expires_at"],
                row["fresh_explains"],
            ),
            row,
        )

    def seated(self, token: str | None, tab_id: str | None) -> Principal:
        """The session, once it's shown to hold the seat from this tab: section 16's control."""
        if not self.settings.gate_enabled:
            return self.session(token)
        if not tab_id:
            raise Problem(400, "tab_required", "This request needs a tab ID")
        session_id = session_id_for(token, self._key) if token else None
        try:
            principal, row = self._load(token)
        except DatabaseUnavailable:
            # Postgres is down. A session keeps going until its lease runs out (section 14).
            with self._recent_lock:
                known = self._recent.get((session_id or "", tab_id))
            if known is not None and known[1] > time.monotonic():
                return known[0]
            raise
        if row["seat_tab"] is None or not row["lease_live"]:
            raise Problem(409, "seat_lost", "This session no longer holds the seat")
        if row["seat_tab"] != tab_id:
            raise Problem(
                409,
                "other_tab",
                "WikiGuard is open in another tab",
                "Close the other tab's session to continue here.",
            )
        seated = Principal(
            principal.session_id,
            principal.role,
            principal.csrf_token,
            principal.created_at,
            principal.expires_at,
            principal.fresh_explains,
            tab_id,
        )
        self._remember(seated, tab_id, float(row["lease_left"]))
        return seated

    # --- The gate's endpoints ---

    def status(self, token: str | None) -> dict[str, Any]:
        """Public, so a visitor can see whether to wait. It says nothing about who's signed in."""
        if not self.settings.gate_enabled:
            return {"enabled": False, "state": "free", "retry_after_seconds": None, "mine": False}
        with self.db.tx() as tx:
            self._sweep(tx)
            holder = self._holder(tx)
        if holder is None:
            return {"enabled": True, "state": "free", "retry_after_seconds": None, "mine": False}
        mine = bool(token) and holder["session_id"] == session_id_for(str(token), self._key)
        left = holder["guest_seconds_left"] if holder["role"] == GUEST else None
        return {"enabled": True, "state": "in_use", "retry_after_seconds": left, "mine": mine}

    def claim(
        self, principal: Principal, tab_id: str, take_over: bool, request_id: str | None
    ) -> None:
        """Take the seat from this tab: after a reload, a crash, or to move from another tab."""
        if not self.settings.gate_enabled:
            return
        with self.db.tx() as tx:
            self._sweep(tx)
            holder = self._holder(tx)
            if holder is None:
                self._claim(tx, principal.session_id, principal.role, tab_id)
                return
            if holder["session_id"] == principal.session_id:
                if holder["tab_id"] != tab_id and not take_over:
                    raise Problem(
                        409,
                        "other_tab",
                        "WikiGuard is open in another tab",
                        "Close the other tab's session to continue here.",
                    )
                self._claim(tx, principal.session_id, principal.role, tab_id, force=True)
                return
            if principal.role != USER:
                raise self._seat_taken(holder)
            if not take_over:
                raise Problem(
                    409,
                    "confirm_take_over",
                    "WikiGuard is in use",
                    extra={"holder_role": holder["role"]},
                )
            self._end_session(tx, holder["session_id"], "taken_over")
            self._claim(tx, principal.session_id, USER, tab_id, force=True)
            audit(tx, USER, "gate.take_over", holder["role"], request_id)

    def heartbeat(self, principal: Principal, tab_id: str, active: bool) -> int:
        if not self.settings.gate_enabled:
            return 0
        with self.db.tx() as tx:
            row = tx.one(
                "UPDATE active_session SET lease_expires_at = now() + %s::interval "
                "WHERE session_id = %s AND tab_id = %s AND lease_expires_at > now() "
                "RETURNING seat",
                (self._lease, principal.session_id, tab_id),
            )
            if row is None:
                raise Problem(409, "seat_lost", "This session no longer holds the seat")
            if active:
                tx.run(
                    "UPDATE sessions SET last_activity_at = now() WHERE id = %s",
                    (principal.session_id,),
                )
            seconds = tx.one(
                """
                SELECT GREATEST(0, EXTRACT(EPOCH FROM LEAST(
                    created_at + %s::interval, last_activity_at + %s::interval
                ) - now()))::int AS left_seconds
                FROM sessions WHERE id = %s
                """,
                (self._max, self._idle, principal.session_id),
            )
        return int(seconds["left_seconds"]) if seconds and principal.role == GUEST else 0

    def release(self, principal: Principal, tab_id: str) -> None:
        """From pagehide, so a normal reload doesn't wait out the lease."""
        if not self.settings.gate_enabled:
            return
        with self.db.tx() as tx:
            tx.run(
                "DELETE FROM active_session WHERE session_id = %s AND tab_id = %s",
                (principal.session_id, tab_id),
            )
