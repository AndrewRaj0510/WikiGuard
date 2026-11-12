"""API fuzzing from the OpenAPI schema (Schemathesis), against the real app and real Postgres.

Schemathesis reads the schema the API publishes and sends each route inputs the schema allows
and inputs it forbids: wrong types, huge numbers, odd Unicode, missing fields. Whatever arrives,
the API must answer with a clean error and never crash, and a documented answer must have the
documented shape. A 503 the API wrote on purpose isn't a crash (scripts/fuzz_checks.py).
security-scan.yml runs the same tool against staging over real HTTP.
"""

from typing import Any

import psycopg
import pytest
import schemathesis
import wikiguard_api.app as api_app
from api_test_helpers import Browser, Stack, seed_edit
from fuzz_checks import never_crashes
from hypothesis import HealthCheck, settings
from schemathesis.specs.openapi.checks import (
    content_type_conformance,
    response_schema_conformance,
)

pytestmark = pytest.mark.integration

# The schema lists only each route's success and its 422. Refusals (401, 403, 404, 429) are
# written by one handler in one format, and test_api.py checks them, so status codes aren't
# compared with the schema here.
CHECKS = (never_crashes, content_type_conformance, response_schema_conformance)
# These end or move the session, which would turn every later request into a 401.
SESSION_ROUTES = "^/v1/(auth/(guest|login|logout)|gate/(claim|release))$"
EXAMPLES_PER_ROUTE = 40
NO_LIMIT = (1_000_000, 60)

schema = schemathesis.pytest.from_fixture("api_schema")


@pytest.fixture
def api_schema(stack: Stack, monkeypatch: pytest.MonkeyPatch) -> Any:
    # Rate limits would answer most of a fuzz run with 429 before any real code ran.
    for name in vars(api_app):
        if name.startswith("LIMIT_"):
            monkeypatch.setattr(api_app, name, NO_LIMIT)
    return schemathesis.openapi.from_asgi("/v1/openapi.json", stack.app)


@pytest.fixture
def signed_in(stack: Stack, db: psycopg.Connection[Any]) -> Browser:
    for rev_id in (1, 2, 3):
        seed_edit(db, rev_id, score=0.2 * rev_id)
    db.commit()
    browser = stack.browser()
    assert browser.login().status_code == 200
    return browser


@schema.exclude(path_regex=SESSION_ROUTES).parametrize()
@settings(
    max_examples=EXAMPLES_PER_ROUTE,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture, HealthCheck.too_slow],
)
def test_no_input_breaks_a_signed_in_route(case: schemathesis.Case, signed_in: Browser) -> None:
    case.call_and_validate(session=signed_in.client, headers=signed_in.headers(), checks=CHECKS)


@schema.parametrize()
@settings(
    max_examples=EXAMPLES_PER_ROUTE,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture, HealthCheck.too_slow],
)
def test_no_input_breaks_a_route_for_someone_not_signed_in(
    case: schemathesis.Case, stack: Stack
) -> None:
    browser = stack.browser()
    case.call_and_validate(session=browser.client, headers={"x-tab-id": browser.tab}, checks=CHECKS)
