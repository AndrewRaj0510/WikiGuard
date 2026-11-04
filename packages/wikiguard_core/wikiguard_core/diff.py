"""Pull the added and removed words out of the HTML that MediaWiki's action=compare returns.

Vandals control this text, so the parser has to be safe on any input. MediaWiki escapes wikitext,
so a tag in an edit arrives as &lt;tag&gt; and can't change the structure.

This doesn't use Python's html.parser, because that took 24.8 seconds on 20 KB of "<a <a <a",
and far longer as the input grew (F-056). The pattern below stops at the next "<" and has a
fixed length limit, so the work grows in step with the input. Anything that isn't a well-formed
tag is treated as text, and nothing here builds a tree or fetches anything.

A row is one of these (F-054):
- Both sides present: a changed line. Only the <ins> and <del> words changed, and the rest is
  shared context that would otherwise count as added.
- One side present, with diff-empty opposite: the whole line was added or removed.
- A marker cell holding an mw-diff-movedpara link: the line moved. Moving text writes nothing new,
  so only words the mover also changed count.
"""

import html
import re
from dataclasses import dataclass, field

MAX_DIFF_HTML_CHARS = 2_000_000
MAX_TEXT_CHARS = 200_000
ADDED = "added"
REMOVED = "removed"

TAG = re.compile(r"<(/?)([a-zA-Z][a-zA-Z0-9]*)([^<>]{0,1000})>")
CLASS_ATTRIBUTE = re.compile(r"""class\s*=\s*["']([^"']*)["']""")


@dataclass(frozen=True)
class DiffText:
    added: str
    removed: str


@dataclass
class _Side:
    whole: list[str] = field(default_factory=list)
    changed: list[str] = field(default_factory=list)
    moved: bool = False

    def text(self, other_side_present: bool) -> str:
        if other_side_present or self.moved:
            return " ".join(part for part in self.changed if part)
        return "".join(self.whole)


class _DiffReader:
    def __init__(self) -> None:
        self.added: list[str] = []
        self.removed: list[str] = []
        self.sides: dict[str, _Side] = {}
        self.current: _Side | None = None
        self.change_depth = 0
        self.change_part: list[str] = []
        self.next_cell_moved = False

    def start_tag(self, tag: str, attributes: str) -> None:
        match = CLASS_ATTRIBUTE.search(attributes)
        classes = match.group(1) if match else ""
        if tag == "tr":
            self.finish_row()
        elif tag == "td":
            self.finish_change()
            self.current = None
            if "diff-addedline" in classes:
                self.open_side(ADDED)
            elif "diff-deletedline" in classes:
                self.open_side(REMOVED)
        elif tag == "a" and "mw-diff-movedpara" in classes:
            self.next_cell_moved = True
        elif self.current is not None and tag in ("ins", "del") and "diffchange" in classes:
            self.change_depth += 1

    def end_tag(self, tag: str) -> None:
        if tag == "tr":
            self.finish_row()
        elif tag == "td":
            self.finish_change()
            self.current = None
        elif tag in ("ins", "del") and self.change_depth:
            self.change_depth -= 1
            if not self.change_depth:
                self.finish_change()

    def text(self, raw: str) -> None:
        if self.current is not None and raw:
            data = html.unescape(raw)
            self.current.whole.append(data)
            if self.change_depth:
                self.change_part.append(data)

    def open_side(self, name: str) -> None:
        self.current = self.sides.setdefault(name, _Side())
        self.current.moved = self.next_cell_moved
        self.next_cell_moved = False

    def finish_change(self) -> None:
        if self.current is not None and self.change_part:
            self.current.changed.append("".join(self.change_part))
        self.change_part = []
        self.change_depth = 0

    def finish_row(self) -> None:
        self.finish_change()
        added, removed = self.sides.get(ADDED), self.sides.get(REMOVED)
        if added:
            self.added.append(added.text(other_side_present=removed is not None))
        if removed:
            self.removed.append(removed.text(other_side_present=added is not None))
        self.sides = {}
        self.current = None
        self.next_cell_moved = False


def parse_diff(diff_html: str) -> DiffText:
    """Added and removed text from a compare response. Any input gives a result, never an error."""
    source = diff_html[:MAX_DIFF_HTML_CHARS]
    reader = _DiffReader()
    position = 0
    for match in TAG.finditer(source):
        reader.text(source[position : match.start()])
        position = match.end()
        closing, name, attributes = match.groups()
        if closing:
            reader.end_tag(name.lower())
        else:
            reader.start_tag(name.lower(), attributes)
    reader.text(source[position:])
    reader.finish_row()
    return DiffText(
        added="\n".join(line for line in reader.added if line)[:MAX_TEXT_CHARS],
        removed="\n".join(line for line in reader.removed if line)[:MAX_TEXT_CHARS],
    )
