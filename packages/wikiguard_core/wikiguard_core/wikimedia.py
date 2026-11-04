"""Polite access to the MediaWiki Action API, shared by the backfill, ingestor, labeler, and purge.

Wikimedia's policy asks for a descriptive User-Agent, serial requests, and respect for maxlag and
Retry-After. Keeping all of that here means no caller can forget it. Errors arrive inside normal
HTTP 200 responses (F-052), so every result is checked for an `error` field.
"""

import random
import time
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

import httpx

MAXLAG_SECONDS = "5"
MAX_ATTEMPTS = 6
BASE_BACKOFF_SECONDS = 1.0
MAX_BACKOFF_SECONDS = 60.0
REQUEST_TIMEOUT_SECONDS = 30.0
# The API accepts up to 50 titles or revision ids per request for normal clients.
BATCH_SIZE = 50
RECENT_CHANGE_PROPS = "ids|title|user|userid|comment|sizes|flags|tags|timestamp|sha1"
# A revision with one of these tags undid something earlier on its page.
REVERTING_TAGS = frozenset({"mw-undo", "mw-rollback", "mw-manual-revert"})
LATER_REVISIONS_TO_CHECK = 50


class WikimediaError(RuntimeError):
    """The API refused the request, or kept failing after every retry."""


class DiffStatus(StrEnum):
    OK = "ok"
    HIDDEN = "hidden"
    MISSING = "missing"
    ERROR = "error"


@dataclass(frozen=True)
class DiffResult:
    status: DiffStatus
    html: str = ""
    error_code: str | None = None


@dataclass(frozen=True)
class RevisionState:
    """What the API says about a revision right now, for labels and for purging."""

    rev_id: int
    tags: frozenset[str] = frozenset()
    missing: bool = False
    content_hidden: bool = False
    comment_hidden: bool = False
    user_hidden: bool = False
    timestamp: str | None = None

    @property
    def has_hidden_content(self) -> bool:
        return self.missing or self.content_hidden or self.comment_hidden or self.user_hidden


def backoff_seconds(attempt: int, rng: Callable[[], float] = random.random) -> float:
    """Exponential with full jitter, so restarting clients don't all retry at the same moment."""
    return float(min(MAX_BACKOFF_SECONDS, BASE_BACKOFF_SECONDS * 2**attempt) * rng())


def format_timestamp(moment: datetime) -> str:
    return moment.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def chunks(items: Iterable[Any], size: int = BATCH_SIZE) -> Iterator[list[Any]]:
    batch: list[Any] = []
    for item in items:
        batch.append(item)
        if len(batch) == size:
            yield batch
            batch = []
    if batch:
        yield batch


class WikimediaClient:
    def __init__(
        self,
        user_agent: str,
        api_url: str = "https://en.wikipedia.org/w/api.php",
        transport: httpx.BaseTransport | None = None,
        sleep: Callable[[float], None] = time.sleep,
        on_request: Callable[[], None] | None = None,
    ) -> None:
        self.api_url = api_url
        self.sleep = sleep
        self.on_request = on_request
        self.request_count = 0
        self.http = httpx.Client(
            headers={"User-Agent": user_agent},
            timeout=REQUEST_TIMEOUT_SECONDS,
            transport=transport,
        )

    def close(self) -> None:
        self.http.close()

    def query(self, **params: str) -> dict[str, Any]:
        """One API call, retried on rate limits, lag, and network failures."""
        full = {"format": "json", "formatversion": "2", "maxlag": MAXLAG_SECONDS, **params}
        last_problem = "no attempt made"
        for attempt in range(MAX_ATTEMPTS):
            self.request_count += 1
            if self.on_request:
                self.on_request()
            try:
                response = self.http.get(self.api_url, params=full)
            except httpx.TransportError as error:
                last_problem = type(error).__name__
                self.sleep(backoff_seconds(attempt))
                continue
            if response.status_code == 429 or response.status_code >= 500:
                last_problem = f"HTTP {response.status_code}"
                self.sleep(self._wait_for(response, attempt))
                continue
            if response.status_code != 200:
                raise WikimediaError(f"HTTP {response.status_code}")
            body: dict[str, Any] = response.json()
            if body.get("error", {}).get("code") == "maxlag":
                last_problem = "maxlag"
                self.sleep(self._wait_for(response, attempt))
                continue
            return body
        raise WikimediaError(f"gave up after {MAX_ATTEMPTS} attempts: {last_problem}")

    @staticmethod
    def _wait_for(response: httpx.Response, attempt: int) -> float:
        retry_after = response.headers.get("Retry-After", "")
        return float(retry_after) if retry_after.isdigit() else backoff_seconds(attempt)

    def compare(self, from_rev: int, to_rev: int) -> DiffResult:
        body = self.query(action="compare", fromrev=str(from_rev), torev=str(to_rev), prop="diff")
        error = body.get("error")
        if error:
            code = error.get("code")
            if code == "missingcontent":
                return DiffResult(DiffStatus.HIDDEN, error_code=code)
            if code in ("nosuchrevid", "missingtitle"):
                return DiffResult(DiffStatus.MISSING, error_code=code)
            return DiffResult(DiffStatus.ERROR, error_code=code)
        return DiffResult(DiffStatus.OK, html=str(body.get("compare", {}).get("body", "")))

    def recent_changes(
        self, start: datetime, limit: int, end: datetime | None = None
    ) -> list[dict[str, Any]]:
        """Up to `limit` in-scope edits, newest first, from `start` going back in time."""
        params = {
            "action": "query",
            "list": "recentchanges",
            "rcstart": format_timestamp(start),
            "rcdir": "older",
            "rctype": "edit",
            "rcnamespace": "0",
            "rcshow": "!bot",
            "rcprop": RECENT_CHANGE_PROPS,
        }
        if end:
            params["rcend"] = format_timestamp(end)
        rows: list[dict[str, Any]] = []
        extra: dict[str, str] = {}
        while len(rows) < limit:
            params["rclimit"] = str(min(500, limit - len(rows)))
            body = self.query(**params, **extra)
            if "error" in body:
                raise WikimediaError(f"recentchanges: {body['error'].get('code')}")
            rows.extend(body.get("query", {}).get("recentchanges", []))
            extra = body.get("continue", {})
            if not extra:
                break
        return rows[:limit]

    def revision_states(self, rev_ids: Iterable[int]) -> dict[int, RevisionState]:
        """Current tags and hidden flags, in batches. Every requested id gets an answer."""
        states: dict[int, RevisionState] = {}
        for batch in chunks(rev_ids):
            params = {
                "action": "query",
                "prop": "revisions",
                "revids": "|".join(str(rev_id) for rev_id in batch),
                "rvprop": "ids|tags|flags|timestamp|user|comment|sha1",
            }
            # A batch can return part of its answer plus a continue marker.
            extra: dict[str, str] = {}
            while True:
                body = self.query(**params, **extra)
                if "error" in body:
                    raise WikimediaError(f"revisions: {body['error'].get('code')}")
                query = body.get("query", {})
                for page in query.get("pages", []):
                    for revision in page.get("revisions", []):
                        states[revision["revid"]] = RevisionState(
                            rev_id=revision["revid"],
                            tags=frozenset(revision.get("tags", [])),
                            content_hidden=bool(revision.get("sha1hidden")),
                            comment_hidden=bool(revision.get("commenthidden")),
                            user_hidden=bool(revision.get("userhidden")),
                            timestamp=revision.get("timestamp"),
                        )
                for key in query.get("badrevids", {}):
                    states[int(key)] = RevisionState(rev_id=int(key), missing=True)
                extra = body.get("continue", {})
                if not extra:
                    break
            # Pages deleted between requests can vanish without appearing in badrevids.
            for rev_id in batch:
                states.setdefault(rev_id, RevisionState(rev_id=rev_id, missing=True))
        return states

    def revision_deletion_ids(self, since: datetime, max_pages: int = 20) -> set[int]:
        """Revisions hidden since a moment, from the public log, without checking each one.

        This finds hidden edits across the whole archive for a few requests. It can't see
        suppression, whose log is private, which is why recent revisions also get re-checked.
        """
        params = {
            "action": "query",
            "list": "logevents",
            "letype": "delete",
            "leaction": "delete/revision",
            "lenamespace": "0",
            "leend": format_timestamp(since),
            "lelimit": "500",
            "leprop": "details|ids|timestamp",
        }
        found: set[int] = set()
        extra: dict[str, str] = {}
        for _ in range(max_pages):
            body = self.query(**params, **extra)
            if "error" in body:
                raise WikimediaError(f"logevents: {body['error'].get('code')}")
            for entry in body.get("query", {}).get("logevents", []):
                details = entry.get("params", {})
                if details.get("type") == "revision":
                    found.update(int(rev_id) for rev_id in details.get("ids", []))
            extra = body.get("continue", {})
            if not extra:
                break
        return found

    def hours_until_next_revert(self, title: str, rev_id: int, edited_at: datetime) -> float | None:
        """Hours from an edit to the next revert on its page, or None if none was found.

        An estimate. The API doesn't say which revert undid which edit, so this takes the first
        revert afterwards. That can only make a delay look shorter, so it errs towards label 1.
        """
        body = self.query(
            action="query",
            prop="revisions",
            titles=title,
            rvstartid=str(rev_id),
            rvdir="newer",
            rvlimit=str(LATER_REVISIONS_TO_CHECK),
            rvprop="ids|timestamp|tags",
        )
        for page in body.get("query", {}).get("pages", []):
            for revision in page.get("revisions", []):
                if revision["revid"] != rev_id and REVERTING_TAGS & set(revision.get("tags", [])):
                    reverted_at = datetime.fromisoformat(revision["timestamp"])
                    return (reverted_at - edited_at).total_seconds() / 3600
        return None

    def user_edit_counts(self, names: Iterable[str]) -> dict[str, int]:
        """Today's edit counts, which is only correct for an edit happening right now (F-007)."""
        counts: dict[str, int] = {}
        for batch in chunks(names):
            body = self.query(
                action="query", list="users", ususers="|".join(batch), usprop="editcount"
            )
            for user in body.get("query", {}).get("users", []):
                if "editcount" in user:
                    counts[user["name"]] = int(user["editcount"])
        return counts

    def edits_before(self, name: str, moment: datetime, cap: int) -> int:
        """How many edits an account had made before a moment, counted up to `cap`."""
        params = {
            "action": "query",
            "list": "usercontribs",
            "ucuser": name,
            "ucstart": format_timestamp(moment),
            "ucdir": "older",
            "ucprop": "ids",
        }
        total = 0
        extra: dict[str, str] = {}
        while total < cap:
            params["uclimit"] = str(min(500, cap - total))
            body = self.query(**params, **extra)
            total += len(body.get("query", {}).get("usercontribs", []))
            extra = body.get("continue", {})
            if not extra:
                break
        return min(total, cap)
