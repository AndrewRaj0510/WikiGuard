from collections.abc import Callable
from typing import Any

from wikiguard_core.identity import is_temporary_account
from wikiguard_core.schemas import (
    SCHEMA_VERSION,
    edit_from_recentchange,
    edit_from_stream,
    is_in_scope_stream_event,
)


def test_only_in_scope_samples_pass_the_filter(stream_rows: list[dict[str, Any]]) -> None:
    assert stream_rows
    for row in stream_rows:
        expected = row["category"].startswith("in_scope")
        assert is_in_scope_stream_event(row["event"]) is expected, row["category"]


def test_stream_event_becomes_an_edit(stream_rows: list[dict[str, Any]], hash_key: str) -> None:
    row = next(r for r in stream_rows if r["category"] == "in_scope_registered")

    edit = edit_from_stream(row["event"], hash_key)

    assert edit.rev_id == row["event"]["revision"]["new"]
    assert edit.parent_rev_id == row["event"]["revision"]["old"]
    assert edit.schema_version == SCHEMA_VERSION
    assert edit.is_anonymous is False
    assert edit.timestamp.tzinfo is not None


def test_stream_temporary_accounts_count_as_logged_out(
    stream_rows: list[dict[str, Any]], hash_key: str
) -> None:
    rows = [r for r in stream_rows if r["category"] == "in_scope_temporary"]
    assert rows

    for row in rows:
        assert is_temporary_account(row["event"]["user"])
        assert edit_from_stream(row["event"], hash_key).is_anonymous is True


def test_api_temp_flag_and_name_pattern_agree(
    load_case: Callable[[str], dict[str, Any]], hash_key: str
) -> None:
    rows = load_case("recentchanges_sample")["query"]["recentchanges"]

    for row in rows:
        by_name = is_temporary_account(row["user"])
        assert by_name == bool(row.get("temp"))
        assert edit_from_recentchange(row, hash_key).is_anonymous == by_name


def test_stream_and_api_shapes_give_the_same_field_types(
    stream_rows: list[dict[str, Any]], load_case: Callable[[str], dict[str, Any]], hash_key: str
) -> None:
    stream = edit_from_stream(
        next(r for r in stream_rows if r["category"] == "in_scope_registered")["event"], hash_key
    )
    api = edit_from_recentchange(
        load_case("recentchanges_sample")["query"]["recentchanges"][0], hash_key
    )

    for name in ("rev_id", "parent_rev_id", "timestamp", "is_minor", "new_length", "old_length"):
        assert type(getattr(stream, name)) is type(getattr(api, name)), name


def test_hidden_rows_parse_with_their_fields_missing(hash_key: str) -> None:
    row = {
        "revid": 5,
        "old_revid": 4,
        "timestamp": "2026-11-02T18:30:00Z",
        "userhidden": True,
        "commenthidden": True,
        "sha1hidden": True,
    }

    edit = edit_from_recentchange(row, hash_key)

    assert edit.user_hash is None
    assert edit.comment is None
    assert (edit.user_hidden, edit.comment_hidden, edit.content_hidden) == (True, True, True)


def test_first_revision_has_no_parent(hash_key: str) -> None:
    row = {"revid": 5, "old_revid": 0, "timestamp": "2026-11-02T18:30:00Z", "user": "A"}

    assert edit_from_recentchange(row, hash_key).parent_rev_id is None
