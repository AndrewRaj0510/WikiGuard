"""Print one version's section of CHANGELOG.md, for the release workflow.

    python scripts/release_notes.py v0.6.0

Fails when the changelog has no section for that version, so a tag can't be released without
notes. The leading "v" is optional.
"""

import argparse
import re
import sys
from pathlib import Path

CHANGELOG = Path(__file__).resolve().parents[1] / "CHANGELOG.md"
HEADING = re.compile(r"^## \[(?P<version>[^\]]+)\]", re.MULTILINE)
# Link definitions at the foot of the file, such as "[0.6.0]: https://...".
LINK = re.compile(r"^\[[^\]]+\]:\s+\S+$", re.MULTILINE)


def section(text: str, version: str) -> str | None:
    """The notes under `## [version]`, up to the next version's heading."""
    wanted = version.removeprefix("v")
    headings = list(HEADING.finditer(text))
    for index, heading in enumerate(headings):
        if heading.group("version") == wanted:
            end = headings[index + 1].start() if index + 1 < len(headings) else len(text)
            body = text[heading.end() : end]
            # Drop the rest of the heading line (the date), then the link definitions.
            body = body.split("\n", 1)[1] if "\n" in body else ""
            return LINK.sub("", body).strip()
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("version")
    args = parser.parse_args()
    notes = section(CHANGELOG.read_text("utf-8"), args.version)
    if not notes:
        print(f"CHANGELOG.md has no notes for {args.version}", file=sys.stderr)  # noqa: T201
        return 1
    print(notes)  # noqa: T201
    return 0


if __name__ == "__main__":
    sys.exit(main())
