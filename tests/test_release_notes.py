"""The release workflow's notes come from CHANGELOG.md, so its format is checked here."""

import re

from release_notes import CHANGELOG, HEADING, section

SAMPLE = """# Changelog

## [Unreleased]

### Added
- Something not out yet.

## [0.2.0] - 2026-02-01

### Added
- The second thing.

## [0.1.0] - 2026-01-01

### Added
- The first thing.

[0.2.0]: https://example.test/compare/v0.1.0...v0.2.0
[0.1.0]: https://example.test/releases/tag/v0.1.0
"""


def test_a_version_gets_only_its_own_notes() -> None:
    assert section(SAMPLE, "v0.2.0") == "### Added\n- The second thing."


def test_the_last_version_stops_before_the_links() -> None:
    assert section(SAMPLE, "0.1.0") == "### Added\n- The first thing."


def test_a_version_with_no_section_has_no_notes() -> None:
    assert section(SAMPLE, "v9.9.9") is None


def test_the_real_changelog_has_an_unreleased_section_first() -> None:
    versions = [match.group("version") for match in HEADING.finditer(CHANGELOG.read_text("utf-8"))]
    assert versions[0] == "Unreleased"


def test_every_released_version_is_dated_numbered_and_newest_first() -> None:
    text = CHANGELOG.read_text("utf-8")
    released = re.findall(r"^## \[(\d+)\.(\d+)\.(\d+)\] - (\d{4}-\d{2}-\d{2})$", text, re.MULTILINE)
    headings = [match.group("version") for match in HEADING.finditer(text)]
    assert len(released) == len(headings) - 1, "a version heading isn't `## [X.Y.Z] - YYYY-MM-DD`"
    numbers = [tuple(int(part) for part in entry[:3]) for entry in released]
    assert numbers == sorted(numbers, reverse=True)
    assert all(section(text, ".".join(entry[:3])) for entry in released)
