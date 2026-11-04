from datetime import UTC, datetime, timedelta
from typing import Any

import httpx
from check_backfill import check_edit_counts, check_exclusions, check_revert_delay
from wikiguard_core.wikimedia import WikimediaClient

EDITED = datetime(2026, 11, 1, 12, tzinfo=UTC)


def row(rev_id: int, **extra: Any) -> dict[str, Any]:
    return {
        "rev_id": rev_id,
        "title": "Moon",
        "timestamp": EDITED,
        "tagged_reverted": 1,
        "diff_status": "ok",
        "user_hidden": False,
        "comment_hidden": False,
        **extra,
    }


def stamp(moment: datetime) -> str:
    return moment.strftime("%Y-%m-%dT%H:%M:%SZ")


def client(handler: Any) -> WikimediaClient:
    return WikimediaClient("t", transport=httpx.MockTransport(handler), sleep=lambda _: None)


def later_revisions(hours_by_rev: dict[int, float]) -> Any:
    """A page history where each edit's revert arrives after the given number of hours."""

    def handler(request: httpx.Request) -> httpx.Response:
        rev_id = int(request.url.params["rvstartid"])
        revisions = [{"revid": rev_id, "timestamp": stamp(EDITED), "tags": ["mw-reverted"]}]
        if rev_id in hours_by_rev:
            revert_time = EDITED + timedelta(hours=hours_by_rev[rev_id])
            revisions.append(
                {"revid": rev_id + 1000, "timestamp": stamp(revert_time), "tags": ["mw-undo"]}
            )
        return httpx.Response(200, json={"query": {"pages": [{"revisions": revisions}]}})

    return handler


def test_measures_how_long_reverts_take() -> None:
    report = check_revert_delay(
        client(later_revisions({1: 0.5, 2: 3.0, 3: 60.0})), [row(1), row(2), row(3)]
    )

    assert report["sampled"] == 3
    assert report["median_hours"] == 3.0
    assert report["within_1_hour"] == 1
    assert report["later_than_48_hours"] == 1
    assert report["late_share"] == round(1 / 3, 4)
    assert report["over_threshold"] is True


def test_threshold_is_not_crossed_when_reverts_are_quick() -> None:
    report = check_revert_delay(client(later_revisions({1: 0.5, 2: 1.0})), [row(1), row(2)])

    assert report["later_than_48_hours"] == 0
    assert report["over_threshold"] is False


def test_rows_with_no_revert_found_are_counted_not_guessed() -> None:
    report = check_revert_delay(client(later_revisions({1: 1.0})), [row(1), row(2)])

    assert report["revert_not_found"] == 1
    assert report["median_hours"] == 1.0


def test_only_usable_reverted_rows_are_sampled() -> None:
    rows = [row(1), row(2, tagged_reverted=0), row(3, diff_status="hidden")]

    report = check_revert_delay(client(later_revisions({1: 1.0})), rows)

    assert report["reverted_rows_available"] == 1


def test_compares_todays_count_with_the_count_at_edit_time() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        params = request.url.params
        if params.get("prop") == "revisions":
            return httpx.Response(
                200,
                json={"query": {"pages": [{"revisions": [{"revid": 1, "user": "~2026-1-1"}]}]}},
            )
        if params.get("list") == "users":
            users = [{"name": "~2026-1-1", "editcount": 150}]
            return httpx.Response(200, json={"query": {"users": users}})
        contributions = [{"revid": number} for number in range(5)]
        return httpx.Response(200, json={"query": {"usercontribs": contributions}})

    report = check_edit_counts(client(handler), [row(1)])

    assert report["compared"] == 1
    assert report["temporary_accounts"] == 1
    assert report["bucket_would_change"] == 1
    assert report["median_extra_edits_since"] == 145


def test_exclusions_are_counted_by_kind() -> None:
    rows = [
        row(1),
        row(2, diff_status="hidden"),
        row(3, diff_status="missing", tagged_reverted=0),
        row(4, comment_hidden=True),
    ]

    report = check_exclusions(rows)

    assert report["excluded"] == 2
    assert report["excluded_share"] == 0.5
    assert report["excluded_that_were_reverted"] == 1
    assert report["comment_hidden_but_usable"] == 1
    assert report["by_status"] == {"ok": 2, "hidden": 1, "missing": 1}
