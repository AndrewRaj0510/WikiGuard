import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from validate_snapshot import REPORT_COLUMNS, main, report_frames
from wikiguard_core.schemas import EditEvent
from wikiguard_core.snapshot import Candidate, build_snapshot, write_snapshot

AS_OF = datetime(2026, 11, 20, 12, tzinfo=UTC)
END = AS_OF - timedelta(hours=48)
NOTEBOOK = Path("notebooks/train_transformers_gpu.ipynb")


def candidate(rev_id: int, days: int, label: int) -> Candidate:
    values: dict[str, Any] = {
        "rev_id": rev_id,
        "parent_rev_id": None,
        "title": "Page",
        "timestamp": END - timedelta(days=days, seconds=rev_id % 3600),
        "is_minor": False,
        "is_anonymous": False,
        "user_hash": "user",
        "comment": None,
        "old_length": 1,
        "new_length": 2,
        "added_text": f"secret text {rev_id}",
    }
    return Candidate(rev_id, EditEvent(**values), label)


def folder(tmp_path: Path, test_positives: int) -> Path:
    rows = [candidate(index, 20, int(index % 10 == 0)) for index in range(1, 200)]
    rows += [candidate(index, 10, int(index % 10 == 0)) for index in range(200, 400)]
    rows += [
        candidate(index, 1 + index % 3, int(index % 10 < 1 and index < 400 + test_positives * 10))
        for index in range(400, 7000)
    ]
    return write_snapshot(build_snapshot(rows, AS_OF, "synthetic"), tmp_path)


def run(arguments: list[str], monkeypatch: pytest.MonkeyPatch) -> int:
    monkeypatch.setattr("sys.argv", ["validate_snapshot.py", *arguments])
    return main()


def test_a_good_snapshot_exits_zero_and_saves_the_results(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    good = folder(tmp_path, 600)

    code = run([str(good), "--out", str(tmp_path / "reports"), "--skip-evidently"], monkeypatch)

    assert code == 0
    saved = json.loads((tmp_path / "reports" / good.name / "validation.json").read_text("utf-8"))
    assert saved["passed"] is True


def test_a_bad_snapshot_exits_one(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    bad = folder(tmp_path, 100)

    code = run([str(bad), "--out", str(tmp_path / "reports"), "--skip-evidently"], monkeypatch)

    assert code == 1
    saved = json.loads((tmp_path / "reports" / bad.name / "validation.json").read_text("utf-8"))
    assert saved["passed"] is False


def test_no_edit_text_can_reach_the_evidently_report(tmp_path: Path) -> None:
    pytest.importorskip("pandas")
    assert not {"added_text", "removed_text", "comment"} & set(REPORT_COLUMNS)

    train, test = report_frames(folder(tmp_path, 600))

    assert "added_text" not in train.columns
    assert "secret text" not in train.to_csv()
    assert "secret text" not in test.to_csv()


def test_notebook_holds_no_outputs_and_no_secret_values() -> None:
    notebook = json.loads(NOTEBOOK.read_text("utf-8"))
    code = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]

    assert code
    assert all(cell["outputs"] == [] and cell["execution_count"] is None for cell in code)
    joined = "".join("".join(cell["source"]) for cell in code)
    assert 'PASSWORD"] = "' not in joined
    assert "trainer.train" in joined
    # "Run all" must stay a test: a full run takes hours and uploads about 630 MB per model.
    assert "FULL_RUN = False" in joined
