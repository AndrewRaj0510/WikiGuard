"""Turn the raw captures in data/fixtures_raw/ into public fixtures in tests/fixtures/wikimedia/.

The repo is public, so every field is treated as unsafe unless a rule below says otherwise. An
unknown field gets its words replaced, which means a new API field can't leak by accident.

Each case has a treatment, set when it was captured:

- ordinary: a settled, unflagged edit. Its title and text stay, so parser tests see real wikitext.
  Usernames, IPs, and emails still go. Its real revision id stays, as the CC BY-SA source link.
- neutral: anything that may be vandalism or was hidden, and every stream event. The HTML and the
  field formats stay, but words become xxxx, titles become placeholders, and ids are remapped, so
  the repo doesn't republish the text or point at the page it was on.

Run it from the repo root after scripts/capture_fixtures.py:

    python scripts/scrub_fixtures.py
"""

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any

RAW_DIR = Path("data/fixtures_raw")
OUT_DIR = Path("tests/fixtures/wikimedia")
STREAM_SAMPLES = {"in_scope_registered": 12, "in_scope_temporary": 12}
STREAM_SAMPLES_DEFAULT = 5
FIRST_FAKE_ID = 900_000_000
SHORTEST_NAME_TO_SEARCH = 3
# A person read each kept-text fixture, and these lost their text as a result. The filters can't
# judge this: none of these edits was reverted or flagged, but their text is about living people.
NEUTRALIZED_AFTER_REVIEW = {
    "compare_removed_only": "the removed text is unverified claims about a living person",
    "compare_small_changed": "the text lists a living person's spouses and children",
    "compare_wikitext_markup": "the text names academy players who may be minors",
}

USER_KEYS = {"user", "fromuser", "touser", "name"}
USER_ID_KEYS = {"userid", "fromuserid", "touserid"}
ID_KEYS = {
    "revid",
    "old_revid",
    "parentid",
    "fromrevid",
    "torevid",
    "rcid",
    "pageid",
    "fromid",
    "toid",
    "logid",
    "logpage",
    "log_id",
    "id",
    "log",
    "ids",
}
TITLE_KEYS = {"title", "fromtitle", "totitle"}
COMMENT_KEYS = {
    "comment",
    "parsedcomment",
    "fromcomment",
    "tocomment",
    "fromparsedcomment",
    "toparsedcomment",
}
URL_KEYS = {"uri", "title_url", "notify_url"}
MESSAGE_KEYS = {"info", "warnings"}
# Machine-written values with nothing an editor chose in them.
VERBATIM_KEYS = {
    "$schema",
    "action",
    "code",
    "contentformat",
    "contentmodel",
    "docref",
    "domain",
    "dt",
    "fromtimestamp",
    "groups",
    "log_action",
    "log_type",
    "registration",
    "request_id",
    "server_name",
    "server_script_path",
    "server_url",
    "servedby",
    "stream",
    "tags",
    "timestamp",
    "topic",
    "totimestamp",
    "type",
    "wiki",
}
USER_PARAMS = {"ususers", "ucuser"}
ID_PARAMS = {"revids", "fromrev", "torev"}

TEMP_NAME = re.compile(r"~(\d{4})-\d+-\d+")
IPV4 = re.compile(r"(?<![\d.])(?:\d{1,3}\.){3}\d{1,3}(?![\d.])")
# Doesn't try to match every compressed IPv6 form, but does match how MediaWiki writes them.
IPV6 = re.compile(r"(?<![\w:])(?:[0-9A-Fa-f]{1,4}:){2,7}(?::?[0-9A-Fa-f]{1,4})+(?![\w:])")
EMAIL = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
USER_REFERENCE = re.compile(
    r"(Special:Contrib(?:ution)?s/|User(?:[ _]talk)?:)([^\]|\"&<>#/\n]+)", re.IGNORECASE
)
# The visible name in [[User:Target|Label]], which can differ from the target.
LINK_LABEL = re.compile(
    r"\[\[(?:Special:Contrib(?:ution)?s/|User(?:[ _]talk)?:)[^\]|]+\|([^\]]+)\]\]"
)
# Fixed wording in those links, as in [[User talk:Target|talk]], rather than anyone's name.
LINK_WORDS = {"talk", "contribs", "contributions"}
HTML_TOKEN = re.compile(r"(<!--.*?-->|<[^>]*>|&#?\w+;)", re.DOTALL)
HTML_ATTRIBUTE = re.compile(r'([\w-]+)="([^"]*)"')
SAFE_HTML_ATTRIBUTES = {"class", "colspan", "data-marker", "dir", "lang"}
# Long pages get a thousands separator: "Line 1,233:".
DIFF_LINE_NUMBER = re.compile(r"Line [\d,]+:")
UUID = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")
# These summaries name other editors in free text ("to last version by ..."), and a link's label
# can differ from its target, so no pattern can pick the names out reliably.
NAMES_PEOPLE = re.compile(r"revert|undid|undo|rollback|restor|Special:Contrib|User", re.IGNORECASE)


def neutralize_text(text: str) -> str:
    """Replace letters and digits, keeping spacing, punctuation, and length."""
    return "".join(
        ("X" if char.isupper() else "x") if char.isalpha() else "0" if char.isdigit() else char
        for char in text
    )


def neutralize_attribute(match: re.Match[str]) -> str:
    name, value = match.group(1), match.group(2)
    if name.lower() in SAFE_HTML_ATTRIBUTES or value.startswith("#"):
        return match.group(0)
    return f'{name}="{neutralize_text(value)}"'


def neutralize_html(html: str) -> str:
    """Replace the words in an HTML fragment. Tags and entities stay for the parser to see."""
    parts = []
    for index, part in enumerate(HTML_TOKEN.split(html)):
        if index % 2 == 0:
            parts.append(part if DIFF_LINE_NUMBER.fullmatch(part) else neutralize_text(part))
        elif part.startswith("<!--"):
            parts.append("<!-- scrubbed -->")
        elif part.startswith("<"):
            parts.append(HTML_ATTRIBUTE.sub(neutralize_attribute, part))
        else:
            parts.append(part)
    return "".join(parts)


class Scrubber:
    """Holds the fake names and ids, so one edit stays linked across every fixture file."""

    def __init__(self, known_names: set[str] | None = None) -> None:
        self.names: dict[str, str] = {}
        self.ids: dict[str, int] = {}
        self.user_ids: dict[int, int] = {}
        self.titles: dict[str, str] = {}
        self.unknown_fields: set[str] = set()
        searchable = {n for n in known_names or set() if len(n) >= SHORTEST_NAME_TO_SEARCH}
        self.known_names = (
            re.compile(
                r"(?<!\w)("
                + "|".join(re.escape(n) for n in sorted(searchable, key=len, reverse=True))
                + r")(?!\w)"
            )
            if searchable
            else None
        )

    def user(self, name: str) -> str:
        """A placeholder that keeps the format, so a temporary account still looks like one."""
        name = name.replace("_", " ").strip()
        if name not in self.names:
            serial = len(self.names) + 1
            temporary = TEMP_NAME.fullmatch(name)
            if temporary:
                self.names[name] = f"~{temporary.group(1)}-{serial:05d}-01"
            elif IPV4.fullmatch(name):
                self.names[name] = f"192.0.2.{serial % 250 + 1}"
            elif IPV6.fullmatch(name):
                self.names[name] = f"2001:db8::{serial:x}"
            else:
                self.names[name] = f"ExampleUser{serial}"
        return self.names[name]

    def number(self, value: int | str) -> int:
        key = str(value)
        if key not in self.ids:
            self.ids[key] = FIRST_FAKE_ID + len(self.ids) + 1
        return self.ids[key]

    def user_id(self, value: int) -> int:
        # Keep 0, which means "no account" in the API.
        if value == 0:
            return 0
        if value not in self.user_ids:
            self.user_ids[value] = 1000 + len(self.user_ids) + 1
        return self.user_ids[value]

    def title(self, title: str) -> str:
        if title not in self.titles:
            self.titles[title] = f"Example article {len(self.titles) + 1}"
        return self.titles[title]

    def clean(self, text: str) -> str:
        """Remove people's identifiers from text that otherwise stays."""
        text = USER_REFERENCE.sub(lambda m: m.group(1) + self.user(m.group(2)), text)
        text = TEMP_NAME.sub(lambda m: self.user(m.group(0)), text)
        if self.known_names:
            text = self.known_names.sub(lambda m: self.user(m.group(1)), text)
        text = IPV4.sub(lambda m: self.user(m.group(0)), text)
        text = IPV6.sub(lambda m: self.user(m.group(0)), text)
        return EMAIL.sub("user@example.org", text)

    def fake_url(self, key: str, url: str) -> str:
        if key == "notify_url":
            base, _, query = url.partition("?")
            return f"{base}?" + re.sub(r"\d+", lambda m: str(self.number(m.group(0))), query)
        base, _, _ = url.partition("/wiki/")
        return f"{base}/wiki/Example_article"

    def replace_known_ids(self, text: str) -> str:
        return re.sub(
            r"\d+",
            lambda m: str(self.ids.get(m.group(0), m.group(0))),
            text,
        )

    def string(self, key: str | None, value: str, neutral: bool) -> str | int:
        if key in USER_KEYS:
            return self.user(value)
        if key in VERBATIM_KEYS:
            return value
        if key == "sha1":
            # A real hash would lead straight to the hidden or reverted revision.
            return hashlib.sha256(value.encode()).hexdigest()[:40] if neutral else value
        if key in ID_KEYS and value.isdigit():
            return self.number(value) if neutral else value
        if key == "id" and UUID.fullmatch(value):
            return value
        if key in TITLE_KEYS:
            return self.title(value) if neutral else self.clean(value)
        if key in COMMENT_KEYS:
            if neutral or NAMES_PEOPLE.search(value):
                return neutralize_html(self.clean(value))
            return self.clean(value)
        if key == "body":
            return neutralize_html(value) if neutral else self.clean(value)
        if key in URL_KEYS:
            return self.fake_url(key, value) if neutral else self.clean(value)
        if key in MESSAGE_KEYS:
            return self.replace_known_ids(value) if neutral else value
        self.unknown_fields.add(str(key))
        return neutralize_text(value)

    def scrub(
        self, value: Any, neutral: bool, key: str | None = None, parent: str | None = None
    ) -> Any:
        if isinstance(value, dict):
            result = {}
            for child_key, child in value.items():
                # badrevids is keyed by the revision id itself.
                out_key = (
                    str(self.number(child_key)) if key == "badrevids" and neutral else child_key
                )
                result[out_key] = self.scrub(child, neutral, child_key, key)
            return result
        if isinstance(value, list):
            return [self.scrub(item, neutral, key, parent) for item in value]
        if isinstance(value, str):
            return self.string(key, value, neutral)
        if isinstance(value, bool) or value is None:
            return value
        if key in USER_ID_KEYS:
            return self.user_id(value)
        is_id = key in ID_KEYS or (parent == "revision" and key in ("old", "new"))
        return self.number(value) if is_id and neutral else value

    def request(self, params: dict[str, str], neutral: bool) -> dict[str, str]:
        """Request parameters are ours, except the people and revisions they name."""
        result = {}
        for key, value in params.items():
            if key in USER_PARAMS:
                result[key] = "|".join(self.user(name) for name in value.split("|"))
            elif key in ID_PARAMS and neutral:
                result[key] = "|".join(str(self.number(rev)) for rev in value.split("|"))
            else:
                result[key] = value
        return result


def real_names(value: Any, key: str | None = None) -> set[str]:
    """Every username in a raw response, so the output can be searched for them afterwards."""
    if isinstance(value, dict):
        return {name for k, v in value.items() for name in real_names(v, k)}
    if isinstance(value, list):
        return {name for item in value for name in real_names(item, key)}
    if isinstance(value, str) and key in USER_KEYS | USER_PARAMS:
        return {part for part in value.split("|") if part}
    if isinstance(value, str) and key in COMMENT_KEYS:
        targets = {match.group(2).strip() for match in USER_REFERENCE.finditer(value)}
        labels = {label.strip() for label in LINK_LABEL.findall(value)}
        return {name for name in targets | labels if name.lower() not in LINK_WORDS}
    return set()


def write_json(path: Path, body: Any) -> None:
    path.write_text(
        json.dumps(body, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n"
    )


def stream_sample(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    counts: dict[str, int] = {}
    sample = []
    for row in rows:
        category = row["category"]
        counts[category] = counts.get(category, 0) + 1
        if counts[category] <= STREAM_SAMPLES.get(category, STREAM_SAMPLES_DEFAULT):
            sample.append(row)
    return sample


def is_neutral(case: dict[str, Any]) -> bool:
    return case["treatment"] != "ordinary" or case["case"] in NEUTRALIZED_AFTER_REVIEW


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    cases = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted((RAW_DIR / "cases").glob("*.json"))
    ]
    # Ordinary cases go first, so their real ids aren't remapped by a neutral case seen earlier.
    cases.sort(key=is_neutral)
    stream_rows = stream_sample(
        [
            json.loads(line)
            for line in (RAW_DIR / "stream_events.jsonl").read_text(encoding="utf-8").splitlines()
        ]
    )

    names = set()
    for case in cases:
        names |= real_names(case["request"]) | real_names(case["response"])
    for row in stream_rows:
        names |= real_names(row["event"])
    scrubber = Scrubber(names)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for stale in OUT_DIR.glob("*.json*"):
        stale.unlink()

    manifest = []
    for case in cases:
        neutral = is_neutral(case)
        treatment = "neutral" if neutral else "ordinary"
        body = {
            "case": case["case"],
            "treatment": treatment,
            "what": case["what"],
            "request": scrubber.request(case["request"], neutral),
            "response": scrubber.scrub(case["response"], neutral),
        }
        write_json(OUT_DIR / f"{case['case']}.json", body)
        entry = {"case": case["case"], "file": f"{case['case']}.json", "treatment": treatment}
        if case["case"] in NEUTRALIZED_AFTER_REVIEW:
            entry["why_neutral"] = NEUTRALIZED_AFTER_REVIEW[case["case"]]
        if not neutral:
            entry["source"] = (
                f"https://en.wikipedia.org/w/index.php?diff={case['request']['torev']}"
            )
        manifest.append(entry)

    scrubbed_stream = [
        {
            "category": row["category"],
            "sse_id": row["sse_id"],
            "event": scrubber.scrub(row["event"], True),
        }
        for row in stream_rows
    ]
    with (OUT_DIR / "stream_events.jsonl").open("w", encoding="utf-8", newline="\n") as file:
        for row in scrubbed_stream:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")
    manifest.append({"case": "stream", "file": "stream_events.jsonl", "treatment": "neutral"})
    write_json(OUT_DIR / "manifest.json", sorted(manifest, key=lambda entry: entry["case"]))

    # Short names stay out of the search, because they match ordinary words.
    leaks = []
    searchable = {name for name in names if len(name) >= SHORTEST_NAME_TO_SEARCH}
    for path in sorted(OUT_DIR.iterdir()):
        if path.suffix in (".json", ".jsonl"):
            text = path.read_text(encoding="utf-8")
            if any(re.search(rf"(?<!\w){re.escape(name)}(?!\w)", text) for name in searchable):
                leaks.append(path.name)
    print(f"Wrote {len(manifest)} fixtures. Searched for {len(searchable)} real usernames.")
    print(f"Fields replaced by the unknown-field rule: {sorted(scrubber.unknown_fields)}")
    if leaks:
        print(f"LEAK: a real username is still in {', '.join(leaks)}")
        return 1
    print("No real username found in the output.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
