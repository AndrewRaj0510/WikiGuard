from collections.abc import Callable
from dataclasses import fields
from datetime import UTC, datetime
from typing import Any

import pytest
from pydantic import ValidationError
from wikiguard_core.diff import DiffText
from wikiguard_core.features import (
    FEATURE_VERSION,
    build_features,
    count_badwords,
    edit_count_bucket,
    normalize_text,
    uppercase_share,
)
from wikiguard_core.schemas import SCHEMA_VERSION, EditEvent, edit_from_recentchange


def make_edit(**changes: Any) -> EditEvent:
    values = {
        "rev_id": 1,
        "parent_rev_id": 0,
        "title": "Moon",
        "timestamp": datetime(2026, 11, 2, 18, 30, tzinfo=UTC),
        "is_minor": False,
        "is_anonymous": False,
        "user_hash": "abc",
        "comment": "fix",
        "old_length": 100,
        "new_length": 150,
    }
    return EditEvent(**{**values, **changes})


def test_metadata_features_come_from_the_edit() -> None:
    features = build_features(make_edit(added_text="SEE https://example.org now"))

    assert features.feature_version == FEATURE_VERSION
    assert features.bytes_delta == 50
    assert features.page_length == 150
    assert features.hour_of_day == 18
    assert features.has_external_link is True
    assert features.chars_added == len("SEE https://example.org now")


def test_uppercase_share_ignores_punctuation_and_handles_no_letters() -> None:
    assert uppercase_share("AB cd!!") == 0.5
    assert uppercase_share("123 !!") == 0.0


@pytest.mark.parametrize(
    ("count", "bucket"),
    [(None, -1), (0, 0), (1, 1), (10, 1), (11, 2), (100, 2), (1000, 3), (1001, 4)],
)
def test_edit_count_buckets(count: int | None, bucket: int) -> None:
    assert edit_count_bucket(count) == bucket


def test_badwords_match_whole_words_only() -> None:
    assert count_badwords("what a stupid IDIOT") == 2
    assert count_badwords("the class assembled") == 0


def test_normalization_undoes_disguises() -> None:
    assert normalize_text("ｓｔｕｐｉｄ") == "stupid"
    assert normalize_text("stu​pid") == "stupid"
    assert normalize_text("stu­pid") == "stupid"
    assert normalize_text("ﬁne") == "fine"
    assert normalize_text("‮gnp.exe") == "gnp.exe"


def test_a_disguised_badword_still_counts() -> None:
    features = build_features(make_edit(added_text="stu​pid ｉｄｉｏｔ"))

    assert features.badword_count == 2


def test_hidden_comment_gives_the_same_features_as_an_empty_one() -> None:
    hidden = build_features(make_edit(comment=None, comment_hidden=True))
    empty = build_features(make_edit(comment=""))

    assert hidden == empty


def test_missing_lengths_do_not_crash() -> None:
    features = build_features(make_edit(old_length=None, new_length=None))

    assert features.bytes_delta == 0
    assert features.page_length == 0


def test_event_rejects_a_tag_field() -> None:
    with pytest.raises(ValidationError):
        make_edit(tags=["mw-reverted"])


def test_event_rejects_an_unknown_schema_version() -> None:
    with pytest.raises(ValidationError):
        make_edit(schema_version=SCHEMA_VERSION + 1)


def test_event_requires_a_time_zone() -> None:
    with pytest.raises(ValidationError):
        make_edit(timestamp=datetime(2026, 11, 2, 18, 30))


def test_no_feature_is_named_after_a_tag_or_label() -> None:
    for field in fields(build_features(make_edit())):
        assert not any(word in field.name for word in ("tag", "label", "revert"))


def test_revert_tags_can_not_change_any_feature(
    load_case: Callable[[str], dict[str, Any]], hash_key: str
) -> None:
    # Real rows, with the tag on and then off, must give identical features.
    rows = load_case("recentchanges_sample")["query"]["recentchanges"]
    reverted = [row for row in rows if "mw-reverted" in row.get("tags", [])]
    assert reverted

    for row in reverted:
        without = {**row, "tags": [tag for tag in row["tags"] if tag != "mw-reverted"]}
        assert build_features(edit_from_recentchange(row, hash_key)) == build_features(
            edit_from_recentchange(without, hash_key)
        )


def test_with_diff_fills_the_text_and_keeps_the_rest() -> None:
    edit = make_edit().with_diff(DiffText(added="new", removed="old"))

    assert edit.added_text == "new"
    assert edit.removed_text == "old"
    assert edit.rev_id == 1
