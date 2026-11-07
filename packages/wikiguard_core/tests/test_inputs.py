import json
from typing import Any

import numpy as np
import pytest
from wikiguard_core.inputs import (
    MAX_SEQUENCE_LENGTH,
    METADATA_CLIP,
    SEPARATOR,
    UNKNOWN,
    MetadataSpec,
    TfidfEncoder,
    TokenVocabulary,
    transformer_text,
)


def row(**changes: Any) -> dict[str, Any]:
    values: dict[str, Any] = {
        "added_text": "hello world",
        "removed_text": "old text",
        "comment": "fixed typo",
        "is_anonymous": False,
        "user_edit_count_bucket": -1,
        "bytes_delta": 10,
        "is_minor": False,
        "chars_added": 11,
        "uppercase_ratio": 0.0,
        "badword_count": 0,
        "has_external_link": False,
        "hour_of_day": 12,
        "page_length": 1000,
    }
    return values | changes


def test_a_feature_that_never_varies_is_left_out() -> None:
    """On the backfill, the edit count is always missing (F-070)."""
    spec = MetadataSpec.fit([row(bytes_delta=1), row(bytes_delta=500)])

    assert "user_edit_count_bucket" not in spec.columns
    assert spec.columns == ("bytes_delta",)


def test_edit_count_is_used_once_training_data_has_it() -> None:
    spec = MetadataSpec.fit([row(user_edit_count_bucket=0), row(user_edit_count_bucket=4)])

    assert "user_edit_count_bucket" in spec.columns


def test_metadata_is_standardized_on_training_values() -> None:
    rows = [row(bytes_delta=1, hour_of_day=3), row(bytes_delta=500, hour_of_day=9)]
    spec = MetadataSpec.fit(rows)

    matrix = spec.transform(rows)

    assert matrix.mean(axis=0) == pytest.approx(0.0, abs=1e-5)
    assert matrix.std(axis=0) == pytest.approx(1.0, abs=1e-5)


def test_an_extreme_edit_can_not_produce_an_extreme_input() -> None:
    spec = MetadataSpec.fit([row(bytes_delta=1), row(bytes_delta=2)])

    matrix = spec.transform([row(bytes_delta=10**9)])

    assert abs(matrix).max() <= METADATA_CLIP


def test_hours_either_side_of_midnight_are_close() -> None:
    rows = [row(hour_of_day=hour) for hour in range(24)]
    spec = MetadataSpec.fit(rows)

    late, early, noon = spec.transform(
        [row(hour_of_day=23), row(hour_of_day=0), row(hour_of_day=12)]
    )

    assert np.linalg.norm(late - early) < np.linalg.norm(late - noon)


def test_metadata_spec_survives_json() -> None:
    spec = MetadataSpec.fit([row(bytes_delta=1), row(bytes_delta=500)])

    assert MetadataSpec.from_dict(json.loads(json.dumps(spec.to_dict()))) == spec


def test_tfidf_separates_added_from_removed_words() -> None:
    rows = [row(added_text="spam", removed_text="fact")] * 3
    encoder = TfidfEncoder.fit(rows, 100, 1, False)

    assert set(encoder.vocabulary) == {"a:spam", "r:fact"}


def test_tfidf_comment_words_are_used_only_when_asked() -> None:
    rows = [row(comment="undid")] * 3

    assert "c:undid" not in TfidfEncoder.fit(rows, 100, 1, False).vocabulary
    assert "c:undid" in TfidfEncoder.fit(rows, 100, 1, True).vocabulary


def test_tfidf_keeps_the_most_common_terms_in_a_fixed_order() -> None:
    rows = [row(added_text="common rare"), row(added_text="common")]

    encoder = TfidfEncoder.fit(rows, 1, 1, False)

    assert list(encoder.vocabulary) == ["a:common"]


def test_tfidf_vectors_are_unit_length_and_ignore_unknown_words() -> None:
    encoder = TfidfEncoder.fit([row(added_text="one two"), row(added_text="one")], 100, 1, False)

    matrix = encoder.dense([row(added_text="one two two never", removed_text="")])

    assert np.linalg.norm(matrix[0]) == pytest.approx(1.0)
    assert encoder.dense([row(added_text="never", removed_text="")]).sum() == 0


def test_tfidf_survives_json() -> None:
    encoder = TfidfEncoder.fit([row(), row(added_text="more words")], 100, 1, True)
    copy = TfidfEncoder.from_dict(json.loads(json.dumps(encoder.to_dict())))

    assert copy.dense([row()]).tolist() == encoder.dense([row()]).tolist()


def test_token_vocabulary_marks_unknown_words_and_the_separator() -> None:
    vocabulary = TokenVocabulary.fit([row(added_text="known", removed_text="")], 100, 1, False)

    ids = vocabulary.encode(row(added_text="known mystery", removed_text=""))

    assert ids.tolist() == [vocabulary.tokens["known"], UNKNOWN, SEPARATOR]


def test_token_sequences_are_never_empty_and_never_too_long() -> None:
    vocabulary = TokenVocabulary.fit([row()], 100, 1, True)

    assert vocabulary.encode(row(added_text="", removed_text="", comment="")).size >= 1
    long = "word " * 5000
    encoded = vocabulary.encode(row(added_text=long, removed_text=long, comment=long))
    assert encoded.size == MAX_SEQUENCE_LENGTH


def test_huge_text_is_cut_before_tokenizing() -> None:
    vocabulary = TokenVocabulary.fit([row()], 100, 1, False)

    assert vocabulary.encode(row(added_text="x " * 200_000)).size <= MAX_SEQUENCE_LENGTH


def test_transformer_text_keeps_the_comment_when_the_addition_is_huge() -> None:
    text = transformer_text(row(added_text="y" * 100_000, comment="short note"))

    assert text.startswith("comment: short note")
    assert len(text) < 1300


def test_a_column_holding_only_rounding_noise_is_left_out(monkeypatch: pytest.MonkeyPatch) -> None:
    # sin(45 degrees) and sin(135 degrees) are exactly equal on Windows and can differ slightly on
    # Linux, so a plain "std > 0" kept this column on one platform only.
    noisy = iter([0.7071067811865476, 0.7071067811865475])
    monkeypatch.setattr(
        "wikiguard_core.inputs._metadata_values",
        lambda _row: {"noise": next(noisy), "real": 1.0 if "seen" in _row else 2.0},
    )

    spec = MetadataSpec.fit([{"seen": 1}, {}])

    assert spec.columns == ("real",)
