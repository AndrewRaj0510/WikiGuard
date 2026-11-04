import json
import logging
from typing import Any

import pytest
from wikiguard_core.diff import DiffText
from wikiguard_core.identity import (
    REDACTED_ACCOUNT,
    REDACTED_IP,
    RedactingFilter,
    hash_user,
    is_ip_address,
    is_logged_out,
    is_temporary_account,
    redact_identifiers,
)
from wikiguard_core.schemas import edit_from_recentchange, edit_from_stream

TEMP = "~2026-81234-55"
IP = "203.0.113.7"


@pytest.mark.parametrize("name", [TEMP, "~2025-1-2"])
def test_temporary_account_names(name: str) -> None:
    assert is_temporary_account(name)
    assert is_logged_out(name)


@pytest.mark.parametrize(
    "name", ["Some Editor", "~Tilde Fan", "~2026", "2026-81234-55", f"x{TEMP}", f"{TEMP}x"]
)
def test_other_names_are_not_temporary_accounts(name: str) -> None:
    assert not is_temporary_account(name)
    assert not is_logged_out(name)


@pytest.mark.parametrize("name", [IP, "2001:db8::1"])
def test_old_ip_usernames_count_as_logged_out(name: str) -> None:
    assert is_ip_address(name)
    assert is_logged_out(name)


def test_a_hex_looking_username_is_not_an_ip_address() -> None:
    # A loose pattern matched a real username in this shape during capture (F-051).
    assert not is_ip_address("Beef42")


def test_anon_flag_counts_as_logged_out() -> None:
    assert is_logged_out("anything", anon_flag=True)


def test_hash_is_keyed_stable_and_not_the_name() -> None:
    first = hash_user("Some Editor", "key-one")

    assert first == hash_user("Some Editor", "key-one")
    assert first == hash_user("Some_Editor", "key-one")
    assert first != hash_user("Some Editor", "key-two")
    assert first != hash_user("Someone Else", "key-one")
    assert "Some" not in first
    assert len(first) == 64


def test_redact_removes_identifiers_and_keeps_the_rest() -> None:
    text = f"Reverted edits by [[Special:Contributions/{TEMP}]] and {IP} or 2001:db8::1 today"

    result = redact_identifiers(text)

    assert TEMP not in result
    assert IP not in result
    assert "2001:db8" not in result
    assert REDACTED_ACCOUNT in result
    assert REDACTED_IP in result
    assert result.startswith("Reverted edits by")
    assert result.endswith("today")


def stream_event(user: str, comment: str) -> dict[str, Any]:
    return {
        "user": user,
        "title": "Moon",
        "comment": comment,
        "timestamp": 1793816364,
        "minor": False,
        "length": {"old": 1, "new": 2},
        "revision": {"old": 1, "new": 2},
    }


@pytest.mark.parametrize("name", [TEMP, IP, "Some Editor"])
def test_no_raw_name_is_returned_from_an_edit(name: str, hash_key: str) -> None:
    from_stream = edit_from_stream(stream_event(name, f"undo {name}"), hash_key)
    from_api = edit_from_recentchange(
        {"revid": 2, "timestamp": "2026-11-02T18:30:00Z", "user": name, "comment": "fine"},
        hash_key,
    )

    for edit in (from_stream, from_api):
        dumped = edit.model_dump_json()
        assert f'"{name}"' not in dumped
        assert name not in json.loads(dumped).values()
        assert name not in repr(edit.user_hash)
    if name != "Some Editor":
        assert name not in from_stream.model_dump_json()


def test_identifiers_in_diff_text_are_removed(hash_key: str) -> None:
    edit = edit_from_stream(stream_event("A", ""), hash_key).with_diff(
        DiffText(added=f"signed {TEMP}", removed=f"from {IP}")
    )

    assert TEMP not in edit.added_text
    assert IP not in edit.removed_text


def test_log_filter_removes_identifiers(caplog: pytest.LogCaptureFixture) -> None:
    logger = logging.getLogger("wikiguard.test.redact")
    logger.addFilter(RedactingFilter())

    with caplog.at_level(logging.INFO, logger=logger.name):
        logger.info("edit by %s from %s", TEMP, IP)

    assert caplog.text
    assert TEMP not in caplog.text
    assert IP not in caplog.text
