"""Ask Wikipedia about every revision in a snapshot, and remove any it has since hidden.

Snapshots are public (DEC-043), so run this before each `dvc push`. Prints counts only. The daily
purge job covers the archive and Postgres, and Phase 5 extends it to snapshots. Until then, this
is the step that keeps a pushed snapshot clean. See README.md for clearing the old version.

    python scripts/recheck_snapshot.py snapshots/NAME
"""

import argparse
import json
import sys
from collections import Counter
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

import pyarrow.parquet as pq
from wikiguard_core.config import ConfigError, WikimediaSettings, load_settings
from wikiguard_core.snapshot import ROWS_FILE, purge_snapshot, verify_snapshot
from wikiguard_core.wikimedia import RevisionState, WikimediaClient


class StateSource(Protocol):
    def revision_states(self, rev_ids: Iterable[int]) -> dict[int, RevisionState]: ...


def recheck(folder: Path, client: StateSource, at: datetime) -> Counter[str]:
    verify_snapshot(folder)
    rev_ids = sorted(pq.read_table(folder / ROWS_FILE, columns=["rev_id"])["rev_id"].to_pylist())
    states = client.revision_states(rev_ids)

    # A revision with no answer can't be assumed safe to publish.
    unanswered = len(set(rev_ids) - states.keys())
    if unanswered:
        raise RuntimeError(f"Wikipedia didn't answer for {unanswered} revisions")

    results: Counter[str] = Counter({"checked": len(rev_ids)})
    for state in states.values():
        results["missing"] += state.missing
        results["content_hidden"] += state.content_hidden
        results["comment_hidden"] += state.comment_hidden
        results["user_hidden"] += state.user_hidden
    hidden = {state.rev_id for state in states.values() if state.has_hidden_content}
    results["rows_removed"] = purge_snapshot(folder, hidden, at)
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    args = parser.parse_args()
    try:
        settings = load_settings(WikimediaSettings)
    except ConfigError as error:
        print(error, file=sys.stderr)
        return 2
    client = WikimediaClient(settings.wikimedia_user_agent, settings.mediawiki_api_url)
    try:
        results = recheck(args.folder, client, datetime.now(UTC))
    finally:
        client.close()
    print(json.dumps(dict(results), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
