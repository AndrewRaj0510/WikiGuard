"""Evidently on real columns. It comes with the trainer's tools, so CI's main job skips this."""

from typing import Any

import numpy as np
import pytest

pytest.importorskip("evidently")

from drift import evidently_drift  # noqa: E402


def rows(
    count: int, seed: int, shift: float = 0.0, text: str = "fixed a typo"
) -> list[dict[str, Any]]:
    rng = np.random.default_rng(seed)
    return [
        {
            "bytes_delta": int(rng.normal(20 + shift * 400, 50)),
            "chars_added": int(abs(rng.normal(60 + shift * 300, 30))),
            "uppercase_ratio": float(rng.beta(1, 9 - shift * 6)),
            "badword_count": int(rng.poisson(0.1 + shift * 2)),
            "hour_of_day": int(rng.integers(0, 24)),
            "page_length": int(abs(rng.normal(20_000, 5_000))),
            "user_edit_count_bucket": -1,
            "is_anonymous": bool(rng.random() < 0.3 + shift * 0.5),
            "is_minor": bool(rng.random() < 0.2),
            "has_external_link": bool(rng.random() < 0.05),
            "added_text": f"{text} {index % 50}",
        }
        for index in range(count)
    ]


def test_the_same_distribution_shows_little_drift() -> None:
    result = evidently_drift(rows(600, 1), rows(600, 2))
    assert result["metadata_share_drifted"] < 0.3
    assert result["text_drift"] is False
    # Backfilled references have no edit count, so it can't be compared (F-122).
    assert result["not_comparable"] == ["user_edit_count_bucket"]


def test_a_shifted_sample_drifts_on_features_and_text() -> None:
    result = evidently_drift(
        rows(600, 1), rows(600, 2, shift=1.0, text="BUY CHEAP FOLLOWERS http://spam.example")
    )
    assert result["metadata_share_drifted"] >= 0.3
    assert result["text_drift"] is True
