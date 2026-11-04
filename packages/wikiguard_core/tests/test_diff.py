import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from wikiguard_core.diff import DiffText, parse_diff


def body(load_case: Callable[[str], dict[str, Any]], case: str) -> str:
    return str(load_case(case)["compare"]["body"])


def test_changed_line_counts_only_the_changed_words(
    load_case: Callable[[str], dict[str, Any]],
) -> None:
    diff = parse_diff(body(load_case, "compare_multi_section"))

    assert diff.added.split("\n") == ["2nd", "2nd", "1st"]
    assert diff.removed.split("\n") == ["2º", "2º", "1º"]


def test_added_lines_keep_the_whole_line_and_remove_nothing(
    load_case: Callable[[str], dict[str, Any]],
) -> None:
    diff = parse_diff(body(load_case, "compare_added_only"))

    assert diff.removed == ""
    assert "City of Buffalo Charter Revision Commission" in diff.added
    assert "Awaiting official results" in diff.added
    assert "=== New York ===" in diff.added


def test_wikitext_comes_back_unescaped(load_case: Callable[[str], dict[str, Any]]) -> None:
    diff = parse_diff(body(load_case, "compare_unicode"))

    assert "<ref>" in diff.added
    assert "&lt;" not in diff.added


def test_non_latin_text_survives(load_case: Callable[[str], dict[str, Any]]) -> None:
    assert "سجادی" in parse_diff(body(load_case, "compare_unicode")).added


def test_moved_text_is_not_counted_as_new(load_case: Callable[[str], dict[str, Any]]) -> None:
    diff = parse_diff(body(load_case, "compare_moved_paragraph"))

    assert "⚫" not in diff.added
    assert "movedpara" not in diff.added
    # The line "|stadium=" only moved, so a whole-line reading would have counted it.
    assert "|stadium=" not in diff.added.split("\n")


def test_every_recorded_diff_parses(compare_cases: list[Path]) -> None:
    parsed = 0
    for path in compare_cases:
        compare = json.loads(path.read_text(encoding="utf-8"))["response"].get("compare")
        if compare:
            diff = parse_diff(compare["body"])
            assert diff.added or diff.removed, path.name
            parsed += 1
    assert parsed >= 8


def test_changed_line_without_inline_marks_adds_nothing() -> None:
    html = (
        '<tr><td class="diff-deletedline"><div>same <del class="diffchange">gone</del></div></td>'
        '<td class="diff-addedline"><div>same</div></td></tr>'
    )

    assert parse_diff(html) == DiffText(added="", removed="gone")


def test_an_edit_can_not_inject_structure_through_escaped_tags() -> None:
    html = (
        '<tr><td class="diff-addedline"><div>'
        '&lt;/td&gt;&lt;td class="diff-deletedline"&gt;x&lt;script&gt;alert(1)&lt;/script&gt;'
        "</div></td></tr>"
    )

    diff = parse_diff(html)

    assert diff.removed == ""
    assert "<script>" in diff.added


def test_empty_and_unrelated_input_give_empty_text() -> None:
    assert parse_diff("") == DiffText("", "")
    assert parse_diff("<p>not a diff</p>") == DiffText("", "")
