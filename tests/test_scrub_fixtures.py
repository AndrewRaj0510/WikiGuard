"""The scrubber's rules, tested on made-up values. No network and no raw captures needed."""

import json

import pytest
from scrub_fixtures import Scrubber, neutralize_html, neutralize_text, real_names


def test_temporary_account_placeholder_keeps_the_format() -> None:
    placeholder = Scrubber().user("~2026-81234-55")

    assert placeholder != "~2026-81234-55"
    assert placeholder.startswith("~2026-")


def test_same_person_gets_the_same_placeholder() -> None:
    scrubber = Scrubber()

    assert scrubber.user("Some Editor") == scrubber.user("Some_Editor")
    assert scrubber.user("Some Editor") != scrubber.user("Someone Else")


@pytest.mark.parametrize(
    ("name", "prefix"), [("203.0.113.7", "192.0.2."), ("2001:4860:4860::8888", "2001:db8::")]
)
def test_ip_usernames_become_documentation_addresses(name: str, prefix: str) -> None:
    assert Scrubber().user(name).startswith(prefix)


def test_clean_removes_usernames_from_a_revert_comment() -> None:
    comment = (
        "Reverted edits by [[Special:Contributions/~2026-81234-55|~2026-81234-55]] "
        "([[User talk:~2026-81234-55|talk]]) to last version by Some Editor"
    )

    cleaned = Scrubber({"Some Editor"}).clean(comment)

    assert "81234" not in cleaned
    assert "Some Editor" not in cleaned
    assert cleaned.startswith("Reverted edits by [[Special:Contributions/~2026-")


def test_clean_removes_user_references_from_parsed_comment_links() -> None:
    comment = '<a href="/wiki/User:Some_Editor" title="User:Some Editor">Some Editor</a>'

    cleaned = Scrubber({"Some Editor"}).clean(comment)

    assert "Some" not in cleaned


def test_clean_removes_ips_and_emails_from_text() -> None:
    cleaned = Scrubber().clean("contact me@mail.example.com from 203.0.113.7 or 2001:4860:4860::1")

    assert "me@mail" not in cleaned
    assert "203.0.113.7" not in cleaned
    assert "2001:4860" not in cleaned


def test_neutralize_text_hides_words_and_digits_but_keeps_shape() -> None:
    assert neutralize_text("Bad wörd, 42! 日本") == "Xxx xxxx, 00! xx"


def test_neutralize_html_keeps_the_markup_a_parser_needs() -> None:
    html = (
        '<tr><td colspan="2" class="diff-lineno">Line 12:</td></tr>'
        '<td class="diff-addedline" data-marker="+"><div>bad &lt;ref&gt; '
        '<ins class="diffchange">word</ins></div></td>'
    )

    result = neutralize_html(html)

    assert '<td colspan="2" class="diff-lineno">Line 12:</td>' in result
    assert 'class="diff-addedline" data-marker="+"' in result
    assert '<ins class="diffchange">xxxx</ins>' in result
    assert "&lt;" in result
    assert "bad" not in result
    assert "word" not in result


def test_neutralize_html_hides_link_targets_and_cache_keys() -> None:
    html = (
        '<a href="/wiki/Some_Page" title="Some Page">x</a><!-- diff cache key enwiki:diff:123 -->'
    )

    result = neutralize_html(html)

    assert "Some" not in result
    assert "123" not in result


def test_unknown_fields_lose_their_words() -> None:
    result = Scrubber().scrub({"brand_new_field": "secret words"}, neutral=False)

    assert result == {"brand_new_field": "xxxxxx xxxxx"}


def test_ordinary_case_keeps_text_and_ids_but_not_people() -> None:
    row = {
        "revid": 123,
        "userid": 77,
        "user": "Some Editor",
        "title": "Moon",
        "comment": "fix typo",
        "body": '<td class="diff-addedline"><div>The Moon</div></td>',
    }

    result = Scrubber({"Some Editor"}).scrub(row, neutral=False)

    assert result["revid"] == 123
    assert result["title"] == "Moon"
    assert result["comment"] == "fix typo"
    assert "The Moon" in result["body"]
    assert result["user"].startswith("ExampleUser")
    assert result["userid"] != 77


def test_neutral_case_replaces_text_ids_and_hash() -> None:
    row = {
        "revid": 123,
        "title": "Moon",
        "comment": "ruined it",
        "sha1": "a" * 40,
        "tags": ["mw-reverted"],
        "newlen": 500,
    }

    result = Scrubber().scrub(row, neutral=True)

    assert result["revid"] != 123
    assert result["title"].startswith("Example article")
    assert result["comment"] == "xxxxxx xx"
    assert result["sha1"] != "a" * 40
    assert len(result["sha1"]) == 40
    assert result["tags"] == ["mw-reverted"]
    assert result["newlen"] == 500


def test_one_revision_keeps_one_fake_id_everywhere() -> None:
    scrubber = Scrubber()

    stream = scrubber.scrub({"revision": {"old": 5, "new": 6}}, neutral=True)
    api = scrubber.scrub({"revid": 6, "old_revid": 5}, neutral=True)

    assert stream["revision"]["new"] == api["revid"]
    assert stream["revision"]["old"] == api["old_revid"]


def test_stream_lengths_are_not_mistaken_for_revision_ids() -> None:
    result = Scrubber().scrub({"length": {"old": 5, "new": 6}}, neutral=True)

    assert result == {"length": {"old": 5, "new": 6}}


def test_flags_named_like_ids_stay_flags() -> None:
    # recentchanges has a boolean "new", and the deletion log has "old" and "new" objects.
    row = {"new": True, "params": {"new": {"bitmask": 3, "content": True}}}

    assert Scrubber().scrub(row, neutral=True) == row


def test_badrevids_keys_are_remapped() -> None:
    result = Scrubber().scrub({"badrevids": {"4000": {"revid": 4000}}}, neutral=True)

    assert "4000" not in json.dumps(result)


def test_request_parameters_lose_people_and_revisions() -> None:
    scrubber = Scrubber()

    result = scrubber.request(
        {"action": "query", "ususers": "Some Editor", "revids": "1|2"}, neutral=True
    )

    assert result["action"] == "query"
    assert "Some Editor" not in result["ususers"]
    assert result["revids"] == f"{scrubber.number(1)}|{scrubber.number(2)}"


def test_kept_text_still_loses_a_revert_summary_with_names_in_free_text() -> None:
    # A real case: the link label held a vandal's name, and the last words another editor's.
    row = {
        "fromcomment": "Reverted edits by [[Special:Contribs/Target|Rude Name]] "
        "([[User talk:Target|talk]]) to last version by Other Editor",
        "tocomment": "/* Loaned out */ Duo",
    }

    result = Scrubber().scrub(row, neutral=False)

    assert "Rude" not in result["fromcomment"]
    assert "Other Editor" not in result["fromcomment"]
    assert result["tocomment"] == "/* Loaned out */ Duo"


def test_real_names_includes_names_found_only_in_comments() -> None:
    raw = {"comment": "Reverted [[Special:Contribs/Target|Rude Name]] and [[User:Plain]]"}

    assert real_names(raw) == {"Target", "Rude Name", "Plain"}


def test_link_wording_is_not_taken_for_a_username() -> None:
    raw = {"comment": "([[User talk:Target|talk]] | [[Special:Contribs/Target|contribs]])"}

    assert real_names(raw) == {"Target"}


def test_event_uuids_keep_their_format_and_other_ids_do_not() -> None:
    uuid = "730ea04b-ee8c-44dd-92c3-3496653be498"

    assert Scrubber().scrub({"id": uuid}, neutral=True) == {"id": uuid}
    assert Scrubber().scrub({"id": 55}, neutral=True) != {"id": 55}


def test_line_numbers_with_a_thousands_separator_survive() -> None:
    html = '<td class="diff-lineno">Line 1,233:</td>'

    assert neutralize_html(html) == html


def test_real_names_finds_every_user_field() -> None:
    raw = {
        "query": {"recentchanges": [{"user": "A Name"}, {"name": "B Name"}]},
        "ususers": "C Name|D Name",
    }

    assert real_names(raw) == {"A Name", "B Name", "C Name", "D Name"}
