"""The edit record every service shares, and how the stream and the API become one.

An EditEvent holds what was known when the edit was made, and nothing later. Its fields are
closed (extra="forbid"), so a revert tag can't reach the features by accident: there's no field
for it, and passing one is an error. Labels travel separately.

The raw username stops here. The ingestor needs it once, to look up the edit count, and after
that only the HMAC is kept, along with is_anonymous.
"""

from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, field_validator

from wikiguard_core.diff import DiffText
from wikiguard_core.identity import hash_user, is_logged_out, redact_identifiers

# Bump when a field changes meaning. Consumers refuse versions they don't know.
SCHEMA_VERSION = 1
ENWIKI = "enwiki"


class EditEvent(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: int = SCHEMA_VERSION
    rev_id: int
    parent_rev_id: int | None
    title: str
    timestamp: datetime
    is_minor: bool
    is_anonymous: bool
    # None when Wikipedia has hidden the value (F-052).
    user_hash: str | None
    comment: str | None
    old_length: int | None
    new_length: int | None
    added_text: str = ""
    removed_text: str = ""
    # The value is measured at edit time on the live path, and missing on backfilled rows.
    user_edit_count: int | None = None
    user_hidden: bool = False
    comment_hidden: bool = False
    content_hidden: bool = False

    @field_validator("schema_version")
    @classmethod
    def check_version(cls, value: int) -> int:
        if value != SCHEMA_VERSION:
            raise ValueError(f"unsupported schema_version {value}, expected {SCHEMA_VERSION}")
        return value

    @field_validator("timestamp")
    @classmethod
    def must_be_utc(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            raise ValueError("timestamp must include a time zone")
        return value.astimezone(UTC)

    def with_diff(self, diff: DiffText) -> "EditEvent":
        return self.model_copy(
            update={
                "added_text": redact_identifiers(diff.added),
                "removed_text": redact_identifiers(diff.removed),
            }
        )

    def with_edit_count(self, count: int | None) -> "EditEvent":
        return self.model_copy(update={"user_edit_count": count})


class ScoreRecord(BaseModel):
    """One model's score for one edit, on the edits.scored topic."""

    model_config = ConfigDict(frozen=True, extra="forbid", protected_namespaces=())

    schema_version: int = SCHEMA_VERSION
    rev_id: int
    model_name: str
    model_version: str
    score: float
    used_fallback: bool = False
    scored_at: datetime


class LabelRecord(BaseModel):
    """The final answer for one edit, on the labels.final topic.

    Hidden and deleted revisions get a reason instead of a label, so they're counted rather than
    silently dropped (the plan's censored-edit rule).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: int = SCHEMA_VERSION
    rev_id: int
    label: int | None
    excluded_reason: str | None = None
    decided_at: datetime
    # Downtime delayed the decision, so a revert after 48 hours may be counted as within it.
    decided_late: bool = False


def is_in_scope_stream_event(event: dict[str, Any]) -> bool:
    """English Wikipedia, article edits, no bots. Filter on `wiki`, never the domain (F-053)."""
    return (
        event.get("wiki") == ENWIKI
        and event.get("type") == "edit"
        and event.get("namespace") == 0
        and event.get("bot") is False
    )


def _clean_comment(comment: str | None) -> str | None:
    return None if comment is None else redact_identifiers(comment)


def edit_from_stream(event: dict[str, Any], hash_key: str) -> EditEvent:
    """A recentchange event from EventStreams. The stream has no tags and no temp flag."""
    user = event["user"]
    return EditEvent(
        rev_id=event["revision"]["new"],
        parent_rev_id=event["revision"].get("old"),
        title=event["title"],
        timestamp=datetime.fromtimestamp(event["timestamp"], UTC),
        is_minor=bool(event.get("minor", False)),
        is_anonymous=is_logged_out(user),
        user_hash=hash_user(user, hash_key),
        comment=_clean_comment(event.get("comment")),
        old_length=event.get("length", {}).get("old"),
        new_length=event.get("length", {}).get("new"),
    )


def edit_from_recentchange(row: dict[str, Any], hash_key: str) -> EditEvent:
    """A list=recentchanges row. Hiding removes a field and sets a flag, so all may be absent."""
    user = row.get("user")
    return EditEvent(
        rev_id=row["revid"],
        parent_rev_id=row.get("old_revid") or None,
        title=row.get("title", ""),
        timestamp=datetime.fromisoformat(row["timestamp"]),
        is_minor=bool(row.get("minor", False)),
        is_anonymous=bool(row.get("temp")) or (user is not None and is_logged_out(user)),
        user_hash=None if user is None else hash_user(user, hash_key),
        comment=_clean_comment(row.get("comment")),
        old_length=row.get("oldlen"),
        new_length=row.get("newlen"),
        user_hidden=bool(row.get("userhidden", False)),
        comment_hidden=bool(row.get("commenthidden", False)),
        content_hidden=bool(row.get("sha1hidden", False)),
    )
