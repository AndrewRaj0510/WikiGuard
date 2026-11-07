"""A tiny synthetic dataset, so training tests run in seconds with no download or network."""

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from wikiguard_core.schemas import EditEvent
from wikiguard_core.snapshot import Candidate, build_snapshot, write_snapshot

# Training tools live in .venv-train (F-049), so the main environment and CI skip these tests.
try:
    import torch  # noqa: F401
except ImportError:
    collect_ignore_glob = ["test_*.py"]

AS_OF = datetime(2026, 11, 20, 12, tzinfo=UTC)
END = AS_OF - timedelta(hours=48)
BAD_WORDS = ("garbage nonsense spam", "stupid junk rubbish", "fake hoax lies")
GOOD_WORDS = ("added a citation", "fixed a small typo", "updated the population figure")


def _edit(rev_id: int, at: datetime, bad: bool) -> EditEvent:
    words = BAD_WORDS if bad else GOOD_WORDS
    values: dict[str, Any] = {
        "rev_id": rev_id,
        "parent_rev_id": None,
        "title": "Page",
        "timestamp": at,
        "is_minor": rev_id % 5 == 0,
        "is_anonymous": bad and rev_id % 2 == 0,
        "user_hash": f"user-{rev_id}",
        "comment": "" if bad else "copyedit",
        "old_length": 1000,
        "new_length": 1000 + (rev_id % 40) - 20,
        "added_text": f"{words[rev_id % 3]} {rev_id}",
        "removed_text": "older words" if rev_id % 2 else "",
    }
    return EditEvent(**values)


def tiny_candidates() -> list[Candidate]:
    rows = []
    rev_id = 0
    for days, count in ((20, 300), (10, 200), (1, 1800)):
        for index in range(count):
            rev_id += 1
            # A clear signal, so a model that works gets a clearly high score. A third positive
            # in test passes the minimum-positives check without many rows.
            bad = index % 3 == 0
            at = END - timedelta(days=days) + timedelta(seconds=index * 7)
            rows.append(Candidate(rev_id, _edit(rev_id, at, bad), int(bad)))
    return rows


@pytest.fixture(scope="session")
def snapshot_folder(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("snapshots")
    return write_snapshot(build_snapshot(tiny_candidates(), AS_OF, "synthetic"), root)
