import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

FIXTURES = Path(__file__).parents[3] / "tests" / "fixtures" / "wikimedia"
TEST_HASH_KEY = "x-test-only-x-test-only-x-test-only-x-test-only"


@pytest.fixture
def hash_key() -> str:
    return TEST_HASH_KEY


@pytest.fixture
def load_case() -> Callable[[str], dict[str, Any]]:
    """The response from a recorded case."""

    def load(case: str) -> dict[str, Any]:
        body: dict[str, Any] = json.loads((FIXTURES / f"{case}.json").read_text(encoding="utf-8"))
        response: dict[str, Any] = body["response"]
        return response

    return load


@pytest.fixture
def stream_rows() -> list[dict[str, Any]]:
    lines = (FIXTURES / "stream_events.jsonl").read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines]


@pytest.fixture
def compare_cases() -> list[Path]:
    return sorted(FIXTURES.glob("compare_*.json"))
