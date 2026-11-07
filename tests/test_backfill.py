"""The backfill's sampling, resume, and row-building logic, with Wikimedia faked."""

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import httpx
import pyarrow.parquet as pq
from backfill import (
    build_labels,
    edits_per_second,
    fetch_diffs,
    label_for,
    list_edits,
    plan_slices,
    read_listing,
    summarize,
)
from wikiguard_core.labels import LABEL_WAIT
from wikiguard_core.wikimedia import WikimediaClient

FIXTURES = Path(__file__).parent / "fixtures" / "wikimedia"
HASH_KEY = "x-test-only-x-test-only-x-test-only-x-test-only"
NOW = datetime(2026, 11, 4, tzinfo=UTC)
DIFF = json.loads((FIXTURES / "compare_multi_section.json").read_text(encoding="utf-8"))["response"]
HIDDEN = json.loads((FIXTURES / "compare_hidden.json").read_text(encoding="utf-8"))["response"]


def rc_row(rev_id: int, **extra: Any) -> dict[str, Any]:
    return {
        "revid": rev_id,
        "old_revid": rev_id - 1,
        "title": "Moon",
        "user": "Real Name",
        "timestamp": "2026-11-01T12:00:00Z",
        "comment": "fix",
        "oldlen": 10,
        "newlen": 20,
        "tags": [],
        **extra,
    }


class FakeApi:
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self.rows = rows
        self.compares = 0
        self.lists = 0
        self.revert_lookups = 0
        self.fail_after: int | None = None

    def __call__(self, request: httpx.Request) -> httpx.Response:
        if request.url.params.get("prop") == "revisions":
            self.revert_lookups += 1
            revert_time = {31: "2026-11-01T13:00:00Z", 32: "2026-11-05T12:00:00Z"}
            rev_id = int(request.url.params["rvstartid"])
            revisions = []
            if rev_id in revert_time:
                revisions = [
                    {"revid": rev_id + 1, "timestamp": revert_time[rev_id], "tags": ["mw-undo"]}
                ]
            return httpx.Response(200, json={"query": {"pages": [{"revisions": revisions}]}})
        if request.url.params["action"] == "compare":
            self.compares += 1
            if self.fail_after is not None and self.compares > self.fail_after:
                raise KeyboardInterrupt
            if request.url.params["torev"] == "13":
                return httpx.Response(200, json=HIDDEN)
            return httpx.Response(200, json=DIFF)
        self.lists += 1
        rows, self.rows = self.rows, []
        return httpx.Response(200, json={"query": {"recentchanges": rows}})


def client(api: FakeApi) -> WikimediaClient:
    return WikimediaClient("t", transport=httpx.MockTransport(api), sleep=lambda _: None)


def test_slices_cover_only_settled_edits_and_spread_across_the_window() -> None:
    starts, per_slice = plan_slices(NOW, 100_000)

    assert max(starts) <= NOW - LABEL_WAIT
    assert min(starts) >= NOW - timedelta(days=30)
    assert max(starts) - min(starts) > timedelta(days=26)
    assert len(starts) * per_slice >= 100_000


def test_small_target_uses_fewer_requests() -> None:
    starts, per_slice = plan_slices(NOW, 2000)

    assert len(starts) == 40
    assert per_slice == 50


def test_listing_never_writes_a_raw_username(tmp_path: Path) -> None:
    list_edits(client(FakeApi([rc_row(11)])), tmp_path, 50, HASH_KEY, NOW)

    assert "Real Name" not in (tmp_path / "listing.jsonl").read_text(encoding="utf-8")


def test_listing_resumes_without_refetching(tmp_path: Path) -> None:
    first = FakeApi([rc_row(11)])
    rows = list_edits(client(first), tmp_path, 100, HASH_KEY, NOW)
    second = FakeApi([])

    assert list_edits(client(second), tmp_path, 100, HASH_KEY, NOW) == rows
    assert first.lists == 2
    assert second.lists == 0


def test_partial_slice_is_dropped_and_broken_lines_are_ignored(tmp_path: Path) -> None:
    path = tmp_path / "listing.jsonl"
    done_row = json.dumps({"edit": {"rev_id": 1}, "tags": []})
    partial = json.dumps({"edit": {"rev_id": 2}, "tags": []})
    path.write_text(
        f'{done_row}\n{{"slice_done": 0, "rows": 1, "span_seconds": 0}}\n{partial}\n{{"broken',
        encoding="utf-8",
    )

    done, rows = read_listing(path)

    assert done == {0}
    assert [row["edit"]["rev_id"] for row in rows] == [1]


def listed(tmp_path: Path, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return list_edits(client(FakeApi(rows)), tmp_path, 50, HASH_KEY, NOW)


def test_rows_carry_label_text_and_no_username(tmp_path: Path) -> None:
    rows = listed(tmp_path, [rc_row(11, tags=["mw-reverted"]), rc_row(12)])

    fetch_diffs(client(FakeApi([])), tmp_path, rows)

    table = pq.read_table(tmp_path / "parts").to_pylist()
    by_rev = {row["rev_id"]: row for row in table}
    assert by_rev[11]["tagged_reverted"] == 1
    assert by_rev[12]["tagged_reverted"] == 0
    assert by_rev[11]["added_text"].split("\n") == ["2nd", "2nd", "1st"]
    assert by_rev[11]["user_edit_count"] is None
    assert "Real Name" not in json.dumps(table, default=str)


def test_hidden_rows_are_counted_without_a_request(tmp_path: Path) -> None:
    rows = listed(tmp_path, [rc_row(11, sha1hidden=True), rc_row(13), rc_row(14)])
    api = FakeApi([])

    fetch_diffs(client(api), tmp_path, rows)

    by_rev = {r["rev_id"]: r for r in pq.read_table(tmp_path / "parts").to_pylist()}
    assert by_rev[11]["diff_status"] == "hidden"
    assert by_rev[13]["diff_status"] == "hidden"
    assert by_rev[14]["diff_status"] == "ok"
    assert api.compares == 2


def test_diffs_resume_from_finished_parts(tmp_path: Path, monkeypatch: Any) -> None:
    monkeypatch.setattr("backfill.BATCH_ROWS", 2)
    rows = listed(tmp_path, [rc_row(rev) for rev in (21, 22, 23, 24, 25)])
    interrupted = FakeApi([])
    interrupted.fail_after = 3

    try:
        fetch_diffs(client(interrupted), tmp_path, rows)
    except KeyboardInterrupt:
        pass
    saved = pq.read_table(tmp_path / "parts").num_rows
    assert saved == 3

    finishing = FakeApi([])
    fetch_diffs(client(finishing), tmp_path, rows)

    final = pq.read_table(tmp_path / "parts").column("rev_id").to_pylist()
    assert sorted(final) == [21, 22, 23, 24, 25]
    assert finishing.compares == 2


def test_tagged_edits_get_their_label_from_the_48_hour_rule(tmp_path: Path) -> None:
    rows = listed(
        tmp_path,
        [
            rc_row(31, tags=["mw-reverted"]),  # Reverted after one hour.
            rc_row(32, tags=["mw-reverted"]),  # Reverted after four days.
            rc_row(33, tags=["mw-reverted"]),  # No revert found in the history.
            rc_row(34),
            rc_row(35, sha1hidden=True),
        ],
    )
    fetch_diffs(client(FakeApi([])), tmp_path, rows)
    api = FakeApi([])

    build_labels(client(api), tmp_path)

    labels = {r["rev_id"]: r for r in pq.read_table(tmp_path / "labels_48h.parquet").to_pylist()}
    assert labels[31]["label"] == 1
    assert labels[31]["revert_hours"] == 1.0
    assert labels[32]["label"] == 0
    assert labels[33]["label"] is None
    assert labels[33]["no_label_reason"] == "revert_not_dated"
    assert labels[34]["label"] == 0
    assert labels[35]["label"] is None
    assert labels[35]["no_label_reason"] == "hidden"
    assert api.revert_lookups == 3


def test_label_step_resumes_without_asking_again(tmp_path: Path) -> None:
    rows = listed(tmp_path, [rc_row(31, tags=["mw-reverted"])])
    fetch_diffs(client(FakeApi([])), tmp_path, rows)
    build_labels(client(FakeApi([])), tmp_path)
    again = FakeApi([])

    build_labels(client(again), tmp_path)

    assert again.revert_lookups == 0


def test_label_rule_boundary() -> None:
    usable = {"diff_status": "ok", "tagged_reverted": 1}

    assert label_for(usable, 48.0) == (1, None)
    assert label_for(usable, 48.01) == (0, None)


def test_edit_rate_gives_every_hour_the_same_weight() -> None:
    # A busy hour at 1.5 a second and a quiet one at 0.5 average 1.0. Total rows over total time
    # would say 0.75, because the quiet hour took three times as long to give its rows.
    slices = [
        {"rows": 151, "span_seconds": 100.0},
        {"rows": 151, "span_seconds": 300.0},
        {"rows": 1, "span_seconds": 0.0},
    ]
    assert edits_per_second(slices) == 1.0


def test_summary_reports_rates_exclusions_and_speed(tmp_path: Path) -> None:
    rows = listed(
        tmp_path,
        [
            rc_row(31, tags=["mw-reverted"], timestamp="2026-11-01T12:00:10Z"),
            rc_row(32, tags=["mw-reverted"], timestamp="2026-11-01T12:00:00Z"),
            rc_row(13, tags=["mw-reverted"], timestamp="2026-11-01T11:59:50Z"),
            rc_row(14, timestamp="2026-11-01T11:59:40Z"),
        ],
    )
    fetch_diffs(client(FakeApi([])), tmp_path, rows)
    build_labels(client(FakeApi([])), tmp_path)

    summary = summarize(tmp_path)

    assert summary["rows"] == 4
    assert summary["tagged_reverted"] == 3
    assert summary["excluded_rows"] == 1
    assert summary["diff_status"] == {"ok": 3, "hidden": 1}
    assert summary["labeled_rows"] == 3
    assert summary["positives_48h"] == 1
    assert summary["tagged_but_reverted_after_48h"] == 1
    assert summary["no_label"] == {"hidden": 1}
    assert summary["sampled_edits_per_second"] == 0.1
