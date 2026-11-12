"""Expand, then contract: a migration must not break the code that's already running.

Deploys run the migration first and the new code after it (.github/workflows/cd.yml), so for a
few minutes the old code runs against the new schema. That's safe when a migration only adds
things ("expand"). Removing or renaming something the old code still uses is not, so it has to
wait for a later release, once no running code needs it ("contract"), and say so in its docstring
with a line starting "Contract:". These tests read the migration files. They need no database.
"""

import re
from pathlib import Path

import pytest

VERSIONS = Path(__file__).parents[1] / "migrations" / "versions"
MARKER = re.compile(r"^Contract:\s*\S", re.MULTILINE)
# What old code can trip over: a table or column it reads is gone or renamed, a column changed
# type, or a column it doesn't fill became required.
BREAKING = {
    "drops a table": re.compile(r"\bDROP\s+TABLE\b", re.IGNORECASE),
    "drops a column": re.compile(r"\bDROP\s+COLUMN\b", re.IGNORECASE),
    "renames something": re.compile(r"\bRENAME\s+(COLUMN|TO)\b", re.IGNORECASE),
    "changes a column's type": re.compile(r"\bALTER\s+COLUMN\s+\w+\s+(SET\s+DATA\s+)?TYPE\b", re.I),
    "makes a column required": re.compile(r"\bSET\s+NOT\s+NULL\b", re.IGNORECASE),
    "drops a column (Alembic)": re.compile(r"\bop\.drop_(table|column)\b"),
    "renames a table (Alembic)": re.compile(r"\bop\.rename_table\b"),
}


def upgrade_source(text: str) -> str:
    """The upgrade function only. A downgrade undoes things, so it drops by design."""
    start = text.index("def upgrade()")
    end = text.find("\ndef ", start + 1)
    return text[start:] if end == -1 else text[start:end]


def breaking_changes(text: str) -> list[str]:
    if MARKER.search(text):
        return []
    body = upgrade_source(text)
    return [name for name, pattern in BREAKING.items() if pattern.search(body)]


MIGRATIONS = sorted(VERSIONS.glob("[0-9]*.py"))


def test_there_are_migrations_to_check() -> None:
    assert len(MIGRATIONS) >= 4


@pytest.mark.parametrize("path", MIGRATIONS, ids=lambda path: path.name)
def test_an_upgrade_only_expands_unless_it_says_it_contracts(path: Path) -> None:
    assert breaking_changes(path.read_text("utf-8")) == []


@pytest.mark.parametrize(
    "statement,found",
    [
        ("ALTER TABLE edits ADD COLUMN note text", []),
        ("CREATE TABLE new_things (id bigint)", []),
        ("ALTER TABLE reviews DROP CONSTRAINT reviews_decision_check", []),
        ("ALTER TABLE edits DROP COLUMN title", ["drops a column"]),
        ("DROP TABLE edits", ["drops a table"]),
        ("ALTER TABLE edits RENAME COLUMN title TO name", ["renames something"]),
        ("ALTER TABLE edits RENAME TO changes", ["renames something"]),
        ("ALTER TABLE edits ALTER COLUMN score TYPE double precision", ["changes a column's type"]),
        ("ALTER TABLE edits ALTER COLUMN note SET NOT NULL", ["makes a column required"]),
    ],
)
def test_the_check_tells_expanding_from_breaking(statement: str, found: list[str]) -> None:
    source = f'def upgrade() -> None:\n    op.execute("{statement}")\n\ndef downgrade() -> None:\n'
    assert breaking_changes(source + '    op.execute("DROP TABLE new_things")\n') == found


def test_a_contract_migration_is_allowed_once_it_says_so() -> None:
    source = (
        '"""Remove the old column.\n\nContract: nothing has read edits.title since v0.7.0.\n"""\n'
        'def upgrade() -> None:\n    op.execute("ALTER TABLE edits DROP COLUMN title")\n'
    )
    assert breaking_changes(source) == []
