from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import httpx
import psycopg
import pytest
from conftest import add_edit
from purge import RECHECK_WINDOW, run
from wikiguard_core.archive import EDITS, read_rows, write_rows
from wikiguard_core.schemas import EditEvent
from wikiguard_core.wikimedia import WikimediaClient

pytestmark = pytest.mark.integration

NOW = datetime(2026, 11, 10, 12, tzinfo=UTC)
RECENT = NOW - timedelta(days=1)


class Api:
    def __init__(
        self, revisions: dict[int, dict[str, Any]] | None = None, logged: list[int] | None = None
    ) -> None:
        self.revisions = revisions or {}
        self.logged = logged or []
        self.asked: set[int] = set()

    def __call__(self, request: httpx.Request) -> httpx.Response:
        params = request.url.params
        if params.get("list") == "logevents":
            entries = [{"params": {"type": "revision", "ids": [str(rev) for rev in self.logged]}}]
            return httpx.Response(200, json={"query": {"logevents": entries}})
        ids = [int(rev) for rev in params["revids"].split("|")]
        self.asked.update(ids)
        found = [{"revid": rev, **self.revisions[rev]} for rev in ids if rev in self.revisions]
        bad = {str(rev): {"missing": True} for rev in ids if rev not in self.revisions}
        return httpx.Response(
            200, json={"query": {"pages": [{"revisions": found}], "badrevids": bad}}
        )


def client(api: Api) -> WikimediaClient:
    return WikimediaClient("t", transport=httpx.MockTransport(api), sleep=lambda _: None)


def archived_edit(rev_id: int) -> dict[str, Any]:
    return EditEvent(
        rev_id=rev_id,
        parent_rev_id=None,
        title="Moon",
        timestamp=RECENT,
        is_minor=False,
        is_anonymous=False,
        user_hash="hash",
        comment="a comment",
        old_length=1,
        new_length=2,
        added_text="words",
    ).model_dump()


def row(db: psycopg.Connection[Any], rev_id: int) -> tuple[Any, ...] | None:
    return db.execute(
        "SELECT added_text, comment, user_hash, content_hidden, purged_at IS NOT NULL "
        "FROM edits WHERE rev_id = %s",
        (rev_id,),
    ).fetchone()


def add_explanation(db: psycopg.Connection[Any], rev_id: int) -> None:
    db.execute(
        "INSERT INTO explanations (rev_id, prompt_version, llm_model, explanation) "
        "VALUES (%s, '1', 'm', 'quotes the edit')",
        (rev_id,),
    )
    db.commit()


def test_visible_revisions_are_left_alone(db: psycopg.Connection[Any], tmp_path: Path) -> None:
    add_edit(db, 1, RECENT)

    results = run(db, client(Api({1: {"tags": []}})), tmp_path, NOW)

    assert results["hidden"] == 0
    assert row(db, 1) == ("words", "a comment", "hash", False, False)


def test_hidden_content_is_cleared_everywhere(db: psycopg.Connection[Any], tmp_path: Path) -> None:
    add_edit(db, 1, RECENT)
    add_explanation(db, 1)
    write_rows(tmp_path, EDITS, [archived_edit(1), archived_edit(2)])

    results = run(db, client(Api({1: {"sha1hidden": True}})), tmp_path, NOW)

    assert results["cleared_in_database"] == 1
    assert results["cleared_in_archive"] == 1
    assert row(db, 1) == (None, "a comment", "hash", True, True)
    assert db.execute("SELECT count(*) FROM explanations").fetchone() == (0,)
    by_rev = {r["rev_id"]: r for r in read_rows(tmp_path, EDITS)}
    assert by_rev[1]["added_text"] is None
    assert by_rev[2]["added_text"] == "words"


def test_only_the_hidden_part_is_cleared(db: psycopg.Connection[Any], tmp_path: Path) -> None:
    add_edit(db, 1, RECENT)
    add_edit(db, 2, RECENT)

    run(db, client(Api({1: {"commenthidden": True}, 2: {"userhidden": True}})), tmp_path, NOW)

    assert row(db, 1) == ("words", None, "hash", False, True)
    assert row(db, 2) == ("words", "a comment", None, False, True)


def test_deleted_page_clears_everything(db: psycopg.Connection[Any], tmp_path: Path) -> None:
    add_edit(db, 1, RECENT)

    run(db, client(Api({})), tmp_path, NOW)

    assert row(db, 1) == (None, None, None, True, True)


def test_the_deletion_log_reaches_edits_outside_the_recheck_window(
    db: psycopg.Connection[Any], tmp_path: Path
) -> None:
    old = NOW - RECHECK_WINDOW - timedelta(days=5)
    add_edit(db, 1, old)
    add_edit(db, 2, old)
    write_rows(tmp_path, EDITS, [archived_edit(3)])
    api = Api({1: {"sha1hidden": True}, 2: {"tags": []}, 3: {"sha1hidden": True}}, logged=[1, 3])

    results = run(db, client(api), tmp_path, NOW)

    assert api.asked == {1, 3}
    assert results["cleared_in_database"] == 1
    assert results["cleared_in_archive"] == 1
    assert row(db, 1) == (None, "a comment", "hash", True, True)
    assert row(db, 2) == ("words", "a comment", "hash", False, False)


def test_rerun_finds_nothing_new_to_do(db: psycopg.Connection[Any], tmp_path: Path) -> None:
    add_edit(db, 1, RECENT)
    api = Api({1: {"sha1hidden": True}})
    run(db, client(api), tmp_path, NOW)

    api.asked = set()
    results = run(db, client(api), tmp_path, NOW)

    assert api.asked == set()
    assert results["hidden"] == 0


def test_audit_log_records_revisions_and_no_text(
    db: psycopg.Connection[Any], tmp_path: Path
) -> None:
    add_edit(db, 1, RECENT)

    run(db, client(Api({1: {"sha1hidden": True}})), tmp_path, NOW)

    entry = db.execute("SELECT action, details FROM audit_log").fetchone()
    assert entry is not None
    assert entry[0] == "purge.hidden"
    assert entry[1] == {"rev_ids": [1]}


def test_the_maintenance_role_has_every_permission_it_needs(
    db: psycopg.Connection[Any], tmp_path: Path
) -> None:
    add_edit(db, 1, RECENT)
    add_explanation(db, 1)
    db.execute("SET ROLE wikiguard_maintenance")
    try:
        results = run(db, client(Api({1: {"sha1hidden": True}})), tmp_path, NOW)
    finally:
        db.rollback()
        db.execute("RESET ROLE")
    assert results["cleared_in_database"] == 1
