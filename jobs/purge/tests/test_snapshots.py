"""Published snapshots and the golden set lose hidden revisions, and the old version is freed only
once the new pointers are published."""

import json
from collections.abc import Iterable, Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from wikiguard_core.alerts import Alerter
from wikiguard_core.dvcops import DvcRepo
from wikiguard_core.schemas import EditEvent
from wikiguard_core.snapshot import (
    Candidate,
    build_snapshot,
    read_manifest,
    read_snapshot,
    write_snapshot,
)
from wikiguard_core.wikimedia import RevisionState

from snapshots import purge_golden, run

AS_OF = datetime(2026, 11, 20, tzinfo=UTC)


class Wikipedia:
    def __init__(self, hidden: set[int], logged: set[int] | None = None) -> None:
        self.hidden = hidden
        self.logged = logged if logged is not None else hidden
        self.asked: set[int] = set()

    def revision_states(self, rev_ids: Iterable[int]) -> dict[int, RevisionState]:
        ids = list(rev_ids)
        self.asked.update(ids)
        return {rev: RevisionState(rev, content_hidden=rev in self.hidden) for rev in ids}

    def revision_deletion_ids(self, since: datetime) -> set[int]:
        return set(self.logged)


class Runner:
    """dvc and git: records commands, and says the pointers are published unless told not."""

    def __init__(self, published: bool = True) -> None:
        self.calls: list[list[str]] = []
        self.published = published

    def __call__(self, args: Sequence[str], cwd: Path) -> str:
        self.calls.append(list(args))
        if list(args[:2]) == ["git", "status"]:
            return "" if self.published else " M snapshots/x.dvc"
        return ""


def make_repo(tmp: Path) -> Path:
    snapshots = tmp / "snapshots"
    candidates = []
    for rev in range(1, 1201):
        at = AS_OF - timedelta(days=2, hours=1) - timedelta(days=(rev % 20))
        edit = EditEvent(
            rev_id=rev,
            parent_rev_id=None,
            title="t",
            timestamp=at,
            is_minor=False,
            is_anonymous=False,
            user_hash=None,
            comment="c",
            old_length=1,
            new_length=2,
            added_text=f"words {rev}",
        )
        candidates.append(Candidate(rev, edit, int(rev % 2 == 0)))
    folder = write_snapshot(build_snapshot(candidates, AS_OF, "test"), snapshots)
    folder.with_name(f"{folder.name}.dvc").write_text("outs: []\n")
    golden = snapshots / "golden-set-v1"
    golden.mkdir()
    rows = [{"rev_id": rev, "checked_label": 1} for rev in (5, 6, 7)]
    (golden / "golden.json").write_text(json.dumps({"summary": {}, "rows": rows}))
    golden.with_name("golden-set-v1.dvc").write_text("outs: []\n")
    return folder


def test_hidden_rows_go_from_the_snapshot_and_the_golden_set(tmp_path: Path) -> None:
    folder = make_repo(tmp_path)
    before = read_manifest(folder)["content_sha256"]
    runner = Runner(published=False)
    alerter = Alerter("purge")
    results = run(DvcRepo(tmp_path, runner), Wikipedia({6, 10}), AS_OF, False, True, alerter)
    assert results["folders"][folder.name]["rows_removed"] == 2
    assert results["folders"]["golden-set-v1"]["rows_removed"] == 1
    assert not {6, 10} & {row["rev_id"] for row in read_snapshot(folder)}
    manifest = read_manifest(folder)
    assert manifest["purges"][0]["previous_content_sha256"] == before
    golden = json.loads((tmp_path / "snapshots/golden-set-v1/golden.json").read_text())
    assert [row["rev_id"] for row in golden["rows"]] == [5, 7]
    assert golden["summary"]["purges"][0]["rows_removed"] == 1
    # Both changed folders are added again and pushed.
    assert ["dvc", "add", f"snapshots/{folder.name}"] in [
        [part.replace("\\", "/") for part in call] for call in runner.calls
    ]
    assert sum(call[:2] == ["dvc", "push"] for call in runner.calls) == 2
    # The pointers aren't on GitHub yet, so the old version stays and the owner is asked.
    assert results["gc"].startswith("skipped: pointer")
    assert not any(call[:2] == ["dvc", "gc"] for call in runner.calls)
    assert alerter.sent[0].title == "Snapshot pointers to publish"


def test_the_daily_run_asks_only_about_logged_revisions(tmp_path: Path) -> None:
    make_repo(tmp_path)
    wikipedia = Wikipedia(hidden={6}, logged={6, 99_999})
    run(DvcRepo(tmp_path, Runner()), wikipedia, AS_OF, False, False, Alerter("purge"))
    assert wikipedia.asked == {6}


def test_the_full_run_asks_about_every_revision_and_finds_suppression(tmp_path: Path) -> None:
    folder = make_repo(tmp_path)
    # Suppressed: hidden, but not in the public log.
    wikipedia = Wikipedia(hidden={8}, logged=set())
    results = run(DvcRepo(tmp_path, Runner()), wikipedia, AS_OF, True, True, Alerter("purge"))
    # Every revision in the snapshot, and the golden set's, which overlap it here.
    assert wikipedia.asked == set(range(1, 1201))
    assert results["folders"][folder.name]["rows_removed"] == 1


def test_nothing_hidden_changes_nothing_and_gc_runs_when_published(tmp_path: Path) -> None:
    make_repo(tmp_path)
    runner = Runner(published=True)
    alerter = Alerter("purge")
    results = run(DvcRepo(tmp_path, runner), Wikipedia(set()), AS_OF, False, True, alerter)
    assert results["pointers_changed"] == [] and results["gc"] == "done"
    assert alerter.sent == []


def test_gc_stays_off_unless_enabled(tmp_path: Path) -> None:
    make_repo(tmp_path)
    runner = Runner(published=True)
    results = run(DvcRepo(tmp_path, runner), Wikipedia(set()), AS_OF, False, False, Alerter("x"))
    assert results["gc"].startswith("skipped: SNAPSHOT_GC_ENABLED")


def test_an_unanswered_revision_stops_the_run(tmp_path: Path) -> None:
    make_repo(tmp_path)

    class Silent(Wikipedia):
        def revision_states(self, rev_ids: Iterable[int]) -> dict[int, RevisionState]:
            return {}

    with pytest.raises(RuntimeError, match="didn't answer"):
        run(DvcRepo(tmp_path, Runner()), Silent({6}), AS_OF, False, False, Alerter("x"))


def test_the_golden_purge_is_counted_not_listed(tmp_path: Path) -> None:
    folder = tmp_path / "golden"
    folder.mkdir()
    (folder / "golden.json").write_text(
        json.dumps({"summary": {"rows": 2}, "rows": [{"rev_id": 1}, {"rev_id": 2}]})
    )
    assert purge_golden(folder, {2}, AS_OF) == 1
    text = (folder / "golden.json").read_text()
    assert '"rows_removed": 1' in text and '"rev_id": 2' not in text
    assert purge_golden(folder, {2}, AS_OF) == 0
