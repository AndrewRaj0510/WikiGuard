import json
from datetime import UTC, datetime, timedelta
from typing import Any

import httpx
import psycopg
import pytest
from conftest import add_edit
from labeler import LATE_AFTER, decide, run
from wikiguard_core.kafka import LABELS_FINAL
from wikiguard_core.schemas import LabelRecord
from wikiguard_core.wikimedia import RevisionState, WikimediaClient

NOW = datetime(2026, 11, 10, 12, tzinfo=UTC)
JUST_DUE = NOW - timedelta(hours=48, minutes=30)


class Publisher:
    def __init__(self) -> None:
        self.messages: list[tuple[str, str, bytes]] = []
        self.fail_flush = False

    def publish(self, topic: str, key: str, value: bytes) -> None:
        self.messages.append((topic, key, value))

    def flush(self, timeout: float = 30.0) -> None:
        if self.fail_flush:
            raise RuntimeError("redpanda down")


class Api:
    """Answers tag lookups, like prop=revisions, from a table of revisions."""

    def __init__(self, revisions: dict[int, dict[str, Any]]) -> None:
        self.revisions = revisions
        self.requests: list[list[int]] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        ids = [int(rev) for rev in request.url.params["revids"].split("|")]
        self.requests.append(ids)
        found = [{"revid": rev, **self.revisions[rev]} for rev in ids if rev in self.revisions]
        bad = {
            str(rev): {"revid": rev, "missing": True} for rev in ids if rev not in self.revisions
        }
        return httpx.Response(
            200, json={"query": {"pages": [{"revisions": found}], "badrevids": bad}}
        )


def client(api: Api) -> WikimediaClient:
    return WikimediaClient("t", transport=httpx.MockTransport(api), sleep=lambda _: None)


# Unit tests of the decision, which need no database.
def test_reverted_edit_gets_label_1() -> None:
    state = RevisionState(1, tags=frozenset({"mw-reverted"}))

    assert decide(1, JUST_DUE, state, NOW).label == 1


def test_untagged_edit_gets_label_0() -> None:
    assert decide(1, JUST_DUE, RevisionState(1), NOW).label == 0


def test_deleted_revision_is_excluded_not_labeled() -> None:
    record = decide(1, JUST_DUE, RevisionState(1, missing=True), NOW)

    assert record.label is None
    assert record.excluded_reason == "deleted"


def test_hidden_content_is_excluded_even_with_a_revert_tag() -> None:
    state = RevisionState(1, tags=frozenset({"mw-reverted"}), content_hidden=True)
    record = decide(1, JUST_DUE, state, NOW)

    assert record.label is None
    assert record.excluded_reason == "hidden"


def test_hidden_comment_alone_keeps_its_label() -> None:
    state = RevisionState(1, tags=frozenset({"mw-reverted"}), comment_hidden=True)

    assert decide(1, JUST_DUE, state, NOW).label == 1


def test_a_catch_up_label_is_flagged_late() -> None:
    assert not decide(1, JUST_DUE, RevisionState(1), NOW).decided_late
    assert decide(1, NOW - LATE_AFTER - timedelta(minutes=1), RevisionState(1), NOW).decided_late


# Tests against real Postgres.
@pytest.mark.integration
def test_only_edits_over_48_hours_old_get_labels(db: psycopg.Connection[Any]) -> None:
    add_edit(db, 1, JUST_DUE)
    add_edit(db, 2, NOW - timedelta(hours=47))
    api = Api({1: {"tags": ["mw-reverted"]}, 2: {"tags": []}})
    publisher = Publisher()

    results = run(db, client(api), publisher, NOW)

    assert results["label_1"] == 1
    assert api.requests == [[1]]
    assert db.execute("SELECT rev_id, label FROM labels").fetchall() == [(1, 1)]
    record = LabelRecord.model_validate_json(publisher.messages[0][2])
    assert (publisher.messages[0][0], record.rev_id, record.label) == (LABELS_FINAL, 1, 1)


@pytest.mark.integration
def test_rerun_changes_nothing(db: psycopg.Connection[Any]) -> None:
    add_edit(db, 1, JUST_DUE)
    api = Api({1: {"tags": ["mw-reverted"]}})
    run(db, client(api), Publisher(), NOW)

    # The tag has gone, and the edit no longer looks due. The first answer stands.
    second = Publisher()
    results = run(db, client(Api({1: {"tags": []}})), second, NOW)

    assert results == {}
    assert second.messages == []
    assert db.execute("SELECT label FROM labels").fetchall() == [(1,)]


@pytest.mark.integration
def test_catch_up_after_downtime_labels_the_backlog_oldest_first(
    db: psycopg.Connection[Any],
) -> None:
    for rev_id, days in ((3, 3), (1, 9), (2, 6)):
        add_edit(db, rev_id, NOW - timedelta(days=days))
    api = Api({rev_id: {"tags": []} for rev_id in (1, 2, 3)})

    first = run(db, client(api), Publisher(), NOW, limit=2)
    second = run(db, client(api), Publisher(), NOW, limit=2)

    assert first["label_0"] == 2
    assert second["label_0"] == 1
    assert api.requests[0] == [1, 2]
    late = db.execute("SELECT bool_and(decided_late) FROM labels").fetchone()
    assert late == (True,)
    assert first["late"] == 2


@pytest.mark.integration
def test_large_backlogs_batch_at_50(db: psycopg.Connection[Any]) -> None:
    for rev_id in range(1, 121):
        add_edit(db, rev_id, JUST_DUE)
    api = Api({rev_id: {"tags": []} for rev_id in range(1, 121)})

    run(db, client(api), Publisher(), NOW)

    assert [len(batch) for batch in api.requests] == [50, 50, 20]


@pytest.mark.integration
def test_hidden_and_deleted_revisions_are_counted(db: psycopg.Connection[Any]) -> None:
    add_edit(db, 1, JUST_DUE)
    add_edit(db, 2, JUST_DUE)
    publisher = Publisher()

    results = run(db, client(Api({1: {"sha1hidden": True}})), publisher, NOW)

    assert results["hidden"] == 1
    assert results["deleted"] == 1
    assert db.execute(
        "SELECT rev_id, label, excluded_reason FROM labels ORDER BY rev_id"
    ).fetchall() == [(1, None, "hidden"), (2, None, "deleted")]
    assert json.loads(publisher.messages[0][2])["excluded_reason"] == "hidden"


@pytest.mark.integration
def test_labels_stay_uncommitted_if_redpanda_fails(db: psycopg.Connection[Any]) -> None:
    add_edit(db, 1, JUST_DUE)
    publisher = Publisher()
    publisher.fail_flush = True

    with pytest.raises(RuntimeError):
        run(db, client(Api({1: {"tags": []}})), publisher, NOW)
    db.rollback()

    assert db.execute("SELECT count(*) FROM labels").fetchone() == (0,)


@pytest.mark.integration
def test_the_labeler_role_has_every_permission_it_needs(db: psycopg.Connection[Any]) -> None:
    add_edit(db, 1, JUST_DUE)
    db.execute("SET ROLE wikiguard_labeler")
    try:
        results = run(db, client(Api({1: {"tags": []}})), Publisher(), NOW)
    finally:
        db.rollback()
        db.execute("RESET ROLE")
    assert results["label_0"] == 1
