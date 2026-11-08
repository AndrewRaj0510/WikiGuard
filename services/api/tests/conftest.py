"""Fixtures for the API tests. The helpers are in ../api_test_helpers.py."""

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import psycopg
import pytest
from api_test_helpers import Browser, Stack
from fastapi.testclient import TestClient


@pytest.fixture
def stack(
    db: psycopg.Connection[Any], api_url: str, tmp_path: Path, request: pytest.FixtureRequest
) -> Iterator[Stack]:
    overrides = getattr(request, "param", {})
    built = Stack(api_url, tmp_path, **overrides)
    with TestClient(built.app):  # Runs startup and shutdown.
        built.wait_for_models()
        yield built


@pytest.fixture
def guest(stack: Stack) -> Browser:
    browser = stack.browser()
    assert browser.guest().status_code == 200
    return browser


@pytest.fixture
def user(stack: Stack) -> Browser:
    browser = stack.browser()
    assert browser.login().status_code == 200
    return browser
