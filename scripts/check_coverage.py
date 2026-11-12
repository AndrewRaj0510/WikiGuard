"""Fail when a part of the code is tested less than its gate allows.

One overall number would let a well-tested package hide an untested service, so each part has
its own gate: 80% for the shared package, which everything depends on, and 70% for each service
and job (docs/PLAN.md, section 10).

    pytest --cov --cov-report=json:coverage.json
    python scripts/check_coverage.py coverage.json
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

CORE_GATE = 80.0
SERVICE_GATE = 70.0
GATES = {
    "packages/wikiguard_core/wikiguard_core/": CORE_GATE,
    "services/api/wikiguard_api/": SERVICE_GATE,
    "services/ingestor/": SERVICE_GATE,
    "services/scorer/": SERVICE_GATE,
    "services/archiver/": SERVICE_GATE,
    "jobs/labeler/": SERVICE_GATE,
    "jobs/monitor/": SERVICE_GATE,
    "jobs/purge/": SERVICE_GATE,
    "jobs/retention/": SERVICE_GATE,
    "jobs/retrain_poller/": SERVICE_GATE,
}


def measure(report: dict[str, Any]) -> dict[str, tuple[int, int]]:
    """Statements and covered statements for each gated part."""
    totals = dict.fromkeys(GATES, (0, 0))
    for name, entry in report["files"].items():
        path = name.replace("\\", "/")
        for prefix in GATES:
            if path.startswith(prefix):
                statements, covered = totals[prefix]
                summary = entry["summary"]
                totals[prefix] = (
                    statements + summary["num_statements"],
                    covered + summary["covered_lines"],
                )
    return totals


def failures(totals: dict[str, tuple[int, int]]) -> list[str]:
    found = []
    for prefix, (statements, covered) in totals.items():
        if statements == 0:
            # A part that vanished from the report would otherwise pass with nothing measured.
            found.append(f"{prefix}: nothing was measured")
        elif 100 * covered / statements < GATES[prefix]:
            found.append(
                f"{prefix}: {100 * covered / statements:.1f}% is under {GATES[prefix]:.0f}%"
            )
    return found


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("report", type=Path, help="coverage's JSON report")
    args = parser.parse_args()
    totals = measure(json.loads(args.report.read_text("utf-8")))
    for prefix, (statements, covered) in totals.items():
        percent = 100 * covered / statements if statements else 0.0
        print(
            f"{prefix:<44} {covered:>5}/{statements:<5} {percent:5.1f}%  gate {GATES[prefix]:.0f}%"
        )  # noqa: T201
    found = failures(totals)
    for problem in found:
        print(f"FAILED {problem}", file=sys.stderr)  # noqa: T201
    return 1 if found else 0


if __name__ == "__main__":
    sys.exit(main())
