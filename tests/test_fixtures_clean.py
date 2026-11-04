"""Fails if a committed fixture holds something the public repo shouldn't have.

This reads only what is in the repo, so it also catches a fixture that was edited by hand or
written by something other than scripts/scrub_fixtures.py.
"""

import json
import re
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

FIXTURES = Path(__file__).parent / "fixtures" / "wikimedia"
PLACEHOLDER = re.compile(r"ExampleUser\d+|~\d{4}-\d{5}-01|192\.0\.2\.\d+|2001:db8::[0-9a-f]+")
USER_FIELDS = {"user", "fromuser", "touser", "name", "ususers", "ucuser"}
IPV4 = re.compile(r"(?<![\d.])(?:\d{1,3}\.){3}\d{1,3}(?![\d.])")
EMAIL = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
USER_REFERENCE = re.compile(r"(?:Special:Contrib(?:ution)?s/|User(?:[ _]talk)?:)([^\]|\"&<>#/\n]+)")
TEMP_NAME = re.compile(r"~\d{4}-\d+-\d+")
# Letters that aren't the x left by the scrubber, outside markup and placeholders.
HTML_TOKEN = re.compile(r"<!--.*?-->|<[^>]*>|&#?\w+;", re.DOTALL)
SECRET_LIKE = re.compile(r"(?i)(api[_-]?key|password|bearer\s|secret[_-]?key)\s*[:=]")


WORDS_REPLACED_FIELDS = {
    "body",
    "comment",
    "parsedcomment",
    "fromcomment",
    "tocomment",
    "fromparsedcomment",
    "toparsedcomment",
}
COMMENT_FIELDS = WORDS_REPLACED_FIELDS - {"body"}
NAMES_PEOPLE = re.compile(r"revert|undid|undo|rollback|restor|Special:Contrib|User", re.IGNORECASE)


def fixture_files() -> list[Path]:
    return sorted(path for path in FIXTURES.iterdir() if path.suffix in (".json", ".jsonl"))


def load(path: Path) -> Any:
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in text.splitlines()]
    return json.loads(text)


def walk(value: Any, key: str | None = None) -> Iterator[tuple[str | None, Any]]:
    """Every leaf value in a fixture, with the field it sits under."""
    if isinstance(value, dict):
        for child_key, child in value.items():
            yield from walk(child, child_key)
    elif isinstance(value, list):
        for item in value:
            yield from walk(item, key)
    else:
        yield key, value


@pytest.fixture(params=fixture_files(), ids=lambda path: path.name)
def fixture_file(request: pytest.FixtureRequest) -> Path:
    return request.param


def test_every_user_field_holds_a_placeholder(fixture_file: Path) -> None:
    for key, value in walk(load(fixture_file)):
        if key in USER_FIELDS and isinstance(value, str):
            for name in value.split("|"):
                assert PLACEHOLDER.fullmatch(name), f"{key} isn't a placeholder"


def test_no_real_ip_address(fixture_file: Path) -> None:
    for address in IPV4.findall(fixture_file.read_text(encoding="utf-8")):
        # Neutral text turns a version number like 2.0.1.15 into 0.0.0.00.
        if set(address) != {"0", "."}:
            assert address.startswith("192.0.2."), f"unexpected address in {fixture_file.name}"


def test_no_email_address(fixture_file: Path) -> None:
    for address in EMAIL.findall(fixture_file.read_text(encoding="utf-8")):
        assert address == "user@example.org", f"unexpected email in {fixture_file.name}"


def test_user_references_point_at_placeholders(fixture_file: Path) -> None:
    for key, value in walk(load(fixture_file)):
        if isinstance(value, str) and key not in ("what", "file"):
            for name in USER_REFERENCE.findall(value):
                # Neutral text has letters replaced, so xxxx names are safe.
                if not set(name.lower()) <= set("x0 -~"):
                    assert PLACEHOLDER.fullmatch(name.strip()), f"reference in {key} isn't scrubbed"


def test_temporary_account_names_are_placeholders(fixture_file: Path) -> None:
    for name in TEMP_NAME.findall(fixture_file.read_text(encoding="utf-8")):
        # Neutral text turns every digit to 0, which leaves nothing to look up.
        if set(name) != {"~", "0", "-"}:
            assert PLACEHOLDER.fullmatch(name), f"real account in {fixture_file.name}"


def test_nothing_that_looks_like_a_secret(fixture_file: Path) -> None:
    assert not SECRET_LIKE.search(fixture_file.read_text(encoding="utf-8"))


def test_neutral_diff_text_has_no_words() -> None:
    manifest = json.loads((FIXTURES / "manifest.json").read_text(encoding="utf-8"))
    neutral_files = [entry["file"] for entry in manifest if entry["treatment"] == "neutral"]
    for name in neutral_files:
        for key, value in walk(load(FIXTURES / name)):
            if key in WORDS_REPLACED_FIELDS and isinstance(value, str):
                visible = HTML_TOKEN.sub("", value)
                visible = re.sub(r"Line [\d,]+:", "", visible)
                assert not re.search(r"[^\W\dxX_]", visible), f"words in {key} of {name}"


def test_neutral_titles_are_placeholders() -> None:
    manifest = json.loads((FIXTURES / "manifest.json").read_text(encoding="utf-8"))
    for entry in manifest:
        if entry["treatment"] == "neutral":
            for key, value in walk(load(FIXTURES / entry["file"])):
                if key in ("title", "fromtitle", "totitle"):
                    assert re.fullmatch(r"Example article \d+", value), f"real title in {entry}"


def test_no_kept_comment_can_name_another_editor() -> None:
    # Revert summaries name people in free text, which the placeholder checks above can't see.
    manifest = json.loads((FIXTURES / "manifest.json").read_text(encoding="utf-8"))
    for entry in manifest:
        if entry["treatment"] == "ordinary":
            for key, value in walk(load(FIXTURES / entry["file"])):
                if key in COMMENT_FIELDS and isinstance(value, str):
                    visible = HTML_TOKEN.sub("", value)
                    assert not NAMES_PEOPLE.search(visible), f"{key} in {entry['file']}"


def test_manifest_lists_every_fixture_with_a_source_for_kept_text() -> None:
    manifest = json.loads((FIXTURES / "manifest.json").read_text(encoding="utf-8"))
    listed = {entry["file"] for entry in manifest}
    on_disk = {path.name for path in fixture_files()} - {"manifest.json"}

    assert listed == on_disk
    for entry in manifest:
        if entry["treatment"] == "ordinary":
            assert entry["source"].startswith("https://en.wikipedia.org/")


def test_fixtures_stay_under_the_repo_file_size_limit(fixture_file: Path) -> None:
    # .pre-commit-config.yaml refuses files over 1024 KB.
    assert fixture_file.stat().st_size < 1024 * 1024
