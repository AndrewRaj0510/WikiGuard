"""Keep published snapshots and the golden set clean (DEC-043, F-092).

Snapshots are public on DagsHub, so when Wikipedia hides a revision, our copy must go there too:

    recheck -> remove hidden rows -> dvc add -> dvc push -> (pointer published) -> dvc gc --cloud

    python snapshots.py           # daily: revisions in Wikipedia's public deletion log
    python snapshots.py --full    # weekly: every revision, which also catches suppression

The daily run is cheap because the deletion log covers the whole 30 days in a few requests. The
log can't show suppression, whose log is private, so the weekly run asks about every revision
(about 1,900 batched requests for 95,000 rows, sent one at a time as Wikimedia asks).

What this job can't do is publish the changed pointer files: that needs a git push, which is the
owner's step (rule 17, and the cluster holds no write token for GitHub). Until the pointers are
published, gc refuses to run (wikiguard_core.dvcops), so the old version stays on DagsHub and the
job alerts every day until it's done. Removed rows are counted in the manifest, never listed.
"""

import argparse
import json
import logging
import sys
from collections import Counter
from collections.abc import Iterable
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Protocol

import pyarrow.parquet as pq
from wikiguard_core.alerts import Alerter
from wikiguard_core.config import (
    AlertSettings,
    CommonSettings,
    ConfigError,
    StorageSettings,
    WikimediaSettings,
    load_settings,
)
from wikiguard_core.dvcops import DvcError, DvcRepo
from wikiguard_core.identity import RedactingFilter
from wikiguard_core.snapshot import ROWS_FILE, purge_snapshot
from wikiguard_core.wikimedia import RevisionState, WikimediaClient

logger = logging.getLogger("purge.snapshots")

GOLDEN_FILE = "golden.json"
LOG_LOOKBACK = timedelta(days=3)


class Settings(CommonSettings, WikimediaSettings, StorageSettings, AlertSettings):
    pass


class StateSource(Protocol):
    def revision_states(self, rev_ids: Iterable[int]) -> dict[int, RevisionState]: ...

    def revision_deletion_ids(self, since: datetime) -> set[int]: ...


def rev_ids_of(folder: Path) -> set[int]:
    if (folder / GOLDEN_FILE).is_file():
        golden = json.loads((folder / GOLDEN_FILE).read_text("utf-8"))
        return {int(row["rev_id"]) for row in golden["rows"]}
    return set(pq.read_table(folder / ROWS_FILE, columns=["rev_id"])["rev_id"].to_pylist())


def hidden_among(client: StateSource, rev_ids: set[int]) -> set[int]:
    if not rev_ids:
        return set()
    states = client.revision_states(sorted(rev_ids))
    # A revision with no answer can't be assumed safe to publish.
    unanswered = len(rev_ids - states.keys())
    if unanswered:
        raise RuntimeError(f"Wikipedia didn't answer for {unanswered} revisions")
    return {state.rev_id for state in states.values() if state.has_hidden_content}


def purge_golden(folder: Path, rev_ids: set[int], at: datetime) -> int:
    """The golden set is frozen (DEC-051), except that hidden text must still go. The count is
    recorded in its summary, so a smaller set can be explained."""
    path = folder / GOLDEN_FILE
    golden: dict[str, Any] = json.loads(path.read_text("utf-8"))
    kept = [row for row in golden["rows"] if int(row["rev_id"]) not in rev_ids]
    removed = len(golden["rows"]) - len(kept)
    if not removed:
        return 0
    golden["rows"] = kept
    golden["summary"].setdefault("purges", []).append(
        {"at": at.astimezone(UTC).isoformat(), "rows_removed": removed}
    )
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as file:
        file.write(json.dumps(golden, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)
    return removed


def run(
    repo: DvcRepo,
    client: StateSource,
    at: datetime,
    full: bool,
    gc_enabled: bool,
    alerter: Alerter,
) -> dict[str, Any]:
    logged = None if full else client.revision_deletion_ids(at - LOG_LOOKBACK)
    results: dict[str, Any] = {"mode": "full" if full else "deletion log", "folders": {}}
    changed = []
    for pointer in repo.pointers():
        folder = pointer.with_suffix("")
        if not folder.is_dir():
            repo.pull(pointer)
        rev_ids = rev_ids_of(folder)
        targets = rev_ids if logged is None else rev_ids & logged
        hidden = hidden_among(client, targets)
        if (folder / GOLDEN_FILE).is_file():
            removed = purge_golden(folder, hidden, at)
        else:
            removed = purge_snapshot(folder, hidden, at)
        counts: Counter[str] = Counter(checked=len(targets), rows_removed=removed)
        results["folders"][folder.name] = dict(counts)
        if removed:
            repo.add(folder)
            repo.push(pointer)
            changed.append(pointer.name)
    results["pointers_changed"] = changed
    try:
        results["gc"] = repo.gc_cloud(gc_enabled)
    except DvcError as error:
        results["gc"] = f"failed: {error}"
    if changed or results["gc"].startswith("skipped: pointer"):
        alerter.send(
            "Snapshot pointers to publish",
            f"{', '.join(changed) or 'earlier changes'}: commit and push the .dvc files, then "
            "the next run removes the old version from DagsHub",
        )
    return results


def main() -> int:
    handler = logging.StreamHandler()
    handler.addFilter(RedactingFilter())
    logging.basicConfig(level=logging.INFO, handlers=[handler])
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--full", action="store_true", help="ask about every revision")
    args = parser.parse_args()
    try:
        settings = load_settings(Settings)
    except ConfigError as error:
        logger.error("%s", error)
        return 2
    webhook = settings.discord_webhook_url
    alerter = Alerter("purge", webhook.get_secret_value() if webhook else None)
    client = WikimediaClient(settings.wikimedia_user_agent, settings.mediawiki_api_url)
    try:
        results = run(
            DvcRepo(Path(settings.snapshot_repo_path)),
            client,
            datetime.now(UTC),
            args.full,
            settings.snapshot_gc_enabled,
            alerter,
        )
    finally:
        client.close()
    logger.info("Snapshot purge finished: %s", json.dumps(results))
    return 0


if __name__ == "__main__":
    sys.exit(main())
