"""Vandals write this input, so the parser and features must survive anything, quickly."""

import time
from collections.abc import Callable
from datetime import UTC, datetime

from hypothesis import given, settings
from hypothesis import strategies as st
from wikiguard_core.diff import MAX_TEXT_CHARS, DiffText, parse_diff
from wikiguard_core.features import build_features, normalize_text
from wikiguard_core.identity import redact_identifiers
from wikiguard_core.schemas import EditEvent

TAGS = ["tr", "td", "div", "ins", "del", "a", "script", "br", "table", "!--", "![CDATA["]
CLASSES = ["diff-addedline", "diff-deletedline", "diffchange", "mw-diff-movedpara-left", "x"]
XSS = [
    "<script>alert(1)</script>",
    "<img src=x onerror=alert(1)>",
    '"><svg/onload=alert(1)>',
    "javascript:alert(1)",
    "&lt;script&gt;",
    "&#x3C;script&#x3E;",
    "<!--",
    "]]>",
    "\x00",
    "퟿",
    "‮",
    "​" * 50,
]
LONGEST_ALLOWED_SECONDS = 3.0


def tag() -> st.SearchStrategy[str]:
    return st.builds(
        lambda name, cls, end: f"</{name}>" if end else f'<{name} class="{cls}">',
        st.sampled_from(TAGS),
        st.sampled_from(CLASSES),
        st.booleans(),
    )


markup = st.lists(st.one_of(tag(), st.text(max_size=40), st.sampled_from(XSS)), max_size=60).map(
    "".join
)


def edit_with(text: str, comment: str | None = None) -> EditEvent:
    return EditEvent(
        rev_id=1,
        parent_rev_id=None,
        title="T",
        timestamp=datetime(2026, 11, 2, tzinfo=UTC),
        is_minor=False,
        is_anonymous=False,
        user_hash=None,
        comment=comment,
        old_length=None,
        new_length=None,
        added_text=text,
    )


@settings(max_examples=300, deadline=None)
@given(markup)
def test_parser_never_fails_on_malformed_html(html: str) -> None:
    diff = parse_diff(html)

    assert isinstance(diff, DiffText)
    assert len(diff.added) <= MAX_TEXT_CHARS


@settings(max_examples=300, deadline=None)
@given(st.text(max_size=500), st.one_of(st.none(), st.text(max_size=100)))
def test_features_never_fail_on_any_text(text: str, comment: str | None) -> None:
    features = build_features(edit_with(text, comment))

    assert 0.0 <= features.uppercase_ratio <= 1.0
    assert features.badword_count >= 0


@settings(max_examples=200, deadline=None)
@given(st.text(max_size=300))
def test_normalizing_twice_changes_nothing_new(text: str) -> None:
    once = normalize_text(text)

    assert "​" not in once
    assert "‮" not in once
    assert normalize_text(once).count("​") == 0


@settings(max_examples=200, deadline=None)
@given(st.text(alphabet="~0123456789-.:abcdef ", max_size=300))
def test_redaction_never_fails_and_leaves_no_temporary_account(text: str) -> None:
    result = redact_identifiers(text)

    assert isinstance(result, str)
    assert "~2026-1-1" not in result


def assert_fast(work: Callable[..., object], *args: object) -> None:
    started = time.monotonic()
    work(*args)
    assert time.monotonic() - started < LONGEST_ALLOWED_SECONDS, getattr(work, "__name__", "work")


def test_huge_inputs_finish_quickly() -> None:
    size = 2_000_000
    row = '<tr><td class="diff-addedline"><div>' + "x" * 50 + "</div></td></tr>"
    for html in (
        "<" * size,
        "<a " * (size // 3),
        "&" * size,
        "<!--" * (size // 4),
        "<div>" * (size // 5),
        "<ins class='diffchange'>" * (size // 24),
        row * (size // len(row)),
    ):
        assert_fast(parse_diff, html)


def test_hostile_text_does_not_slow_redaction_or_features() -> None:
    size = 1_000_000
    for text in (
        "~" * size,
        "~2026-" * (size // 6),
        "1." * (size // 2),
        "a:" * (size // 2),
        "1:" * (size // 2),
        "​" * size,
        "a" * size,
    ):
        assert_fast(redact_identifiers, text)
        assert_fast(build_features, edit_with(text))
