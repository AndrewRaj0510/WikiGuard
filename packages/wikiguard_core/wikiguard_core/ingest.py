"""Turning a bare edit into a full one, shared by the live ingestor and the gap repair tool."""

from dataclasses import dataclass

from wikiguard_core.diff import parse_diff
from wikiguard_core.schemas import EditEvent
from wikiguard_core.wikimedia import DiffStatus, WikimediaClient

REASON_MISSING = "missing"
REASON_API_ERROR = "api_error"
REASON_MALFORMED = "malformed"


@dataclass(frozen=True)
class DiffOutcome:
    edit: EditEvent | None
    failure_reason: str | None = None


def attach_diff(client: WikimediaClient, edit: EditEvent) -> DiffOutcome:
    """Fetch and parse the diff. A hidden revision still counts as an edit, just without text."""
    if edit.content_hidden:
        return DiffOutcome(edit)
    if edit.parent_rev_id is None:
        return DiffOutcome(None, REASON_MISSING)
    result = client.compare(edit.parent_rev_id, edit.rev_id)
    if result.status is DiffStatus.OK:
        return DiffOutcome(edit.with_diff(parse_diff(result.html)))
    if result.status is DiffStatus.HIDDEN:
        return DiffOutcome(edit.model_copy(update={"content_hidden": True}))
    if result.status is DiffStatus.MISSING:
        return DiffOutcome(None, REASON_MISSING)
    return DiffOutcome(None, REASON_API_ERROR)
