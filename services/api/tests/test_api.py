"""The API over real Postgres. Marked integration: CI's Postgres job runs these."""

import gzip
import json
import threading
from datetime import UTC, datetime, timedelta
from typing import Any

import psycopg
import pytest
import smoke_test
from api_test_helpers import METRICS_TOKEN, PASSWORD, Browser, Stack, make_settings, one, seed_edit
from fastapi.routing import APIRoute
from psycopg.types.json import Jsonb
from wikiguard_api.access import Principal
from wikiguard_api.app import MAX_PAGE_SIZE, SESSION_COOKIE, create_app
from wikiguard_api.foundations import DatabaseUnavailable

pytestmark = pytest.mark.integration

TAB_B = "tab-bbbbbbbb"


# --- Authorization: every route, found from the app itself, so a new one can't be missed ---

PUBLIC_ROUTES = {
    ("GET", "/health"),
    ("GET", "/ready"),
    ("GET", "/v1/openapi.json"),
    ("POST", "/v1/auth/guest"),
    ("POST", "/v1/auth/login"),
    ("GET", "/v1/gate/status"),
}
# Only `user`. Everything else protected is also open to guests.
USER_ONLY_ROUTES = {
    ("PUT", "/v1/edits/{rev_id}/review"),
    ("POST", "/v1/admin/retrain"),
    ("POST", "/v1/admin/promote"),
    ("POST", "/v1/admin/rollback"),
    ("GET", "/v1/admin/requests"),
    ("GET", "/v1/admin/audit-log"),
}
BODIES = {
    "/v1/admin/retrain": {"reason": "a reason"},
    "/v1/admin/promote": {"reason": "a reason", "model_version": "2"},
    "/v1/admin/rollback": {"reason": "a reason"},
    "/v1/edits/{rev_id}/review": {"decision": "ok"},
    "/v1/gate/claim": {"tab_id": "tab-aaaaaaaa"},
    "/v1/gate/heartbeat": {},
    "/v1/gate/release": {"tab_id": "tab-aaaaaaaa", "csrf_token": "x"},
    "/v1/predict": {"added_text": "x"},
}


def all_routes(stack: Stack) -> list[tuple[str, str]]:
    found = []
    for route in stack.app.routes:
        if isinstance(route, APIRoute):
            found += [(method, route.path) for method in route.methods - {"HEAD", "OPTIONS"}]
    return sorted(found)


def call(browser: Browser, method: str, path: str) -> Any:
    url = path.replace("{rev_id}", "1")
    options = {"json": BODIES[path]} if path in BODIES else {}
    return getattr(browser, method.lower())(url, **options)


def test_the_route_lists_stay_honest(stack: Stack) -> None:
    routes = set(all_routes(stack))
    assert PUBLIC_ROUTES - {("GET", "/v1/openapi.json")} <= routes
    assert USER_ONLY_ROUTES <= routes


def test_every_protected_route_refuses_someone_not_signed_in(stack: Stack) -> None:
    browser = stack.browser()
    protected = [route for route in all_routes(stack) if route not in PUBLIC_ROUTES]
    assert len(protected) > 15
    for method, path in protected:
        expected = {401}
        response = call(browser, method, path)
        assert response.status_code in expected, (method, path, response.status_code)


def test_every_user_only_route_refuses_a_guest(stack: Stack, guest: Browser) -> None:
    for method, path in sorted(USER_ONLY_ROUTES):
        response = call(guest, method, path)
        assert response.status_code == 403, (method, path, response.text)
        assert response.json()["code"] == "forbidden"


def test_a_refused_guest_action_is_audited(
    stack: Stack, guest: Browser, db: psycopg.Connection[Any]
) -> None:
    guest.post("/v1/admin/rollback", json={"reason": "a reason"})
    assert one(db, "SELECT actor, action, target FROM audit_log WHERE action = 'auth.denied'") == (
        "guest",
        "auth.denied",
        "/v1/admin/rollback",
    )
    assert one(db, "SELECT count(*) FROM retrain_requests") == (0,)


def test_user_can_reach_every_user_only_route(
    stack: Stack, user: Browser, db: psycopg.Connection[Any]
) -> None:
    seed_edit(db, 1)
    for method, path in sorted(USER_ONLY_ROUTES):
        response = call(user, method, path)
        assert response.status_code in {200, 202}, (method, path, response.text)


@pytest.mark.parametrize(
    ("method", "path"),
    [
        ("POST", "/v1/predict"),
        ("POST", "/v1/explain/{rev_id}"),
        ("PUT", "/v1/edits/{rev_id}/review"),
        ("POST", "/v1/admin/retrain"),
        ("POST", "/v1/gate/heartbeat"),
        ("POST", "/v1/auth/logout"),
    ],
)
def test_writes_without_the_csrf_token_are_refused(user: Browser, method: str, path: str) -> None:
    user.csrf = "wrong"
    response = call(user, method, path)
    assert response.status_code == 403
    assert response.json()["code"] == "bad_csrf"


def test_writes_from_another_site_are_refused(user: Browser) -> None:
    response = user.post(
        "/v1/predict", json={"added_text": "x"}, headers={"origin": "https://evil.example"}
    )
    assert response.status_code == 403
    assert response.json()["code"] == "bad_origin"


def test_the_listed_origin_is_allowed(user: Browser) -> None:
    response = user.post(
        "/v1/predict", json={"added_text": "x"}, headers={"origin": "http://localhost:3000"}
    )
    assert response.status_code == 200


def test_sign_in_from_another_site_is_refused(stack: Stack) -> None:
    browser = stack.browser()
    response = browser.client.post(
        "/v1/auth/guest",
        json={"tab_id": browser.tab},
        headers={"origin": "https://evil.example"},
    )
    assert response.status_code == 403


def test_a_reused_cookie_is_refused_after_sign_out(stack: Stack, user: Browser) -> None:
    token = user.cookie()
    assert user.post("/v1/auth/logout").status_code == 204
    thief = stack.browser()
    thief.client.cookies.set(SESSION_COOKIE, str(token))
    assert thief.get("/v1/auth/me").status_code == 401


def test_a_made_up_cookie_is_refused(stack: Stack) -> None:
    browser = stack.browser()
    browser.client.cookies.set(SESSION_COOKIE, "made-up")
    assert browser.get("/v1/queue").status_code == 401


def test_the_stored_session_id_isnt_the_cookie(user: Browser, db: psycopg.Connection[Any]) -> None:
    stored = one(db, "SELECT id FROM sessions")[0]
    assert stored != user.cookie()


def test_a_wrong_password_is_refused_and_audited(stack: Stack, db: psycopg.Connection[Any]) -> None:
    browser = stack.browser()
    response = browser.login("wrong password")
    assert response.status_code == 401
    assert "password" not in json.dumps(response.json()).replace("Wrong password", "")
    assert one(db, "SELECT count(*) FROM audit_log WHERE action = 'auth.sign_in_failed'") == (1,)


def test_session_cookie_attributes_in_production(
    db: psycopg.Connection[Any], api_url: str, tmp_path: Any
) -> None:
    built = Stack(api_url, tmp_path, environment="prod")
    built.app.state.context.db.open()
    browser = Browser(built.app, base_url="https://testserver")
    response = browser.guest()
    built.app.state.context.db.close()
    cookie = response.headers["set-cookie"].lower()
    assert "httponly" in cookie
    assert "secure" in cookie
    assert "samesite=lax" in cookie
    assert "strict-transport-security" in response.headers


# --- Login lockout ---


def test_five_failures_lock_that_ip_and_the_right_password_then_fails(
    stack: Stack, db: psycopg.Connection[Any]
) -> None:
    browser = stack.browser()
    for _ in range(5):
        assert browser.login("wrong").status_code == 401
    response = browser.login()
    assert response.status_code == 429
    assert response.json()["code"] == "login_locked"
    assert int(response.headers["retry-after"]) > 0


def test_the_lockout_survives_a_restart(
    stack: Stack, db: psycopg.Connection[Any], api_url: str, tmp_path: Any
) -> None:
    browser = stack.browser()
    for _ in range(5):
        browser.login("wrong")
    restarted = Stack(api_url, tmp_path / "again", make=create_app)
    restarted.app.state.context.db.open()
    response = restarted.browser().login()
    restarted.app.state.context.db.close()
    assert response.status_code == 429


def test_a_success_clears_the_count(stack: Stack, db: psycopg.Connection[Any]) -> None:
    browser = stack.browser()
    for _ in range(4):
        browser.login("wrong")
    assert browser.login().status_code == 200
    assert one(db, "SELECT count(*) FROM login_throttle WHERE scope LIKE 'ip:%%'") == (0,)


def test_the_login_rate_limit_applies_before_any_hashing(stack: Stack) -> None:
    browser = stack.browser()
    responses = [browser.login("wrong").status_code for _ in range(11)]
    assert responses[-1] == 429


# --- Trusted-proxy IP handling ---


@pytest.mark.parametrize("stack", [{"edge_shared_secret": "edge-test-secret"}], indirect=True)
def test_with_an_edge_secret_direct_requests_are_refused(stack: Stack) -> None:
    browser = stack.browser()
    response = browser.client.post("/v1/auth/guest", json={"tab_id": browser.tab})
    assert response.status_code == 403
    assert response.json()["code"] == "direct_access"


@pytest.mark.parametrize("stack", [{"edge_shared_secret": "edge-test-secret"}], indirect=True)
def test_health_stays_reachable_without_the_edge_secret(stack: Stack) -> None:
    assert stack.browser().client.get("/health").status_code == 200


@pytest.mark.parametrize("stack", [{"git_commit": "0123abc"}], indirect=True)
def test_health_names_the_commit_it_was_built_from(stack: Stack) -> None:
    assert stack.browser().client.get("/health").json() == {"status": "ok", "commit": "0123abc"}


def test_health_says_unknown_when_no_commit_is_set(stack: Stack) -> None:
    assert stack.browser().client.get("/health").json()["commit"] == "unknown"


@pytest.mark.parametrize("stack", [{"git_commit": "0123abc"}], indirect=True)
def test_the_deploy_smoke_test_passes_against_the_real_app(stack: Stack) -> None:
    """The smoke test's own tests use a pretend API, which could repeat my assumptions (F-099)."""
    done = smoke_test.run(stack.browser().client, None, "0123abc", 0, lambda _: None)
    assert len(done) == 3


@pytest.mark.parametrize(
    "stack", [{"edge_shared_secret": "edge-test-secret", "git_commit": "0123abc"}], indirect=True
)
def test_the_smoke_test_sees_a_closed_origin_as_closed(stack: Stack) -> None:
    smoke_test.check_origin_closed(stack.browser().client)


def test_a_commit_that_isnt_hex_stops_the_api_starting(stack: Stack) -> None:
    with pytest.raises(ValueError, match="git_commit"):
        make_settings(stack.settings.database_url.get_secret_value(), git_commit="<script>")


@pytest.mark.parametrize("stack", [{"edge_shared_secret": "edge-test-secret"}], indirect=True)
def test_the_workers_client_ip_is_believed_only_with_its_secret(stack: Stack) -> None:
    browser = stack.browser()

    def sign_in(ip: str) -> int:
        return browser.client.post(
            "/v1/auth/guest",
            json={"tab_id": "not valid!"},
            headers={"x-edge-secret": "edge-test-secret", "x-wikiguard-client-ip": ip},
        ).status_code

    assert sign_in("203.0.113.1") == 422
    for ip in ("203.0.113.2", "203.0.113.3"):
        assert sign_in(ip) == 422


def test_spoofed_forwarding_headers_dont_escape_the_limit(stack: Stack) -> None:
    browser = stack.browser()
    statuses = []
    for number in range(8):
        statuses.append(
            browser.client.post(
                "/v1/auth/guest",
                json={"tab_id": browser.tab},
                headers={
                    "x-forwarded-for": f"198.51.100.{number}",
                    "x-real-ip": f"198.51.100.{number}",
                    "x-wikiguard-client-ip": f"198.51.100.{number}",
                    "cf-connecting-ip": f"198.51.100.{number}",
                    "forwarded": f"for=198.51.100.{number}",
                },
            ).status_code
        )
    assert 429 in statuses


# --- The single seat ---


def test_a_second_browser_is_told_to_wait(stack: Stack, guest: Browser) -> None:
    other = stack.browser(TAB_B)
    response = other.guest()
    assert response.status_code == 409
    body = response.json()
    assert body["code"] == "seat_taken"
    assert 0 < body["retry_after_seconds"] <= 300
    assert int(response.headers["retry-after"]) > 0


def test_a_refused_visitor_leaves_nothing_behind(
    stack: Stack, guest: Browser, db: psycopg.Connection[Any]
) -> None:
    stack.browser(TAB_B).guest()
    assert one(db, "SELECT count(*) FROM sessions") == (1,)


def test_simultaneous_sign_ins_seat_exactly_one(stack: Stack, db: psycopg.Connection[Any]) -> None:
    browsers = [stack.browser(f"tab-{index:08d}") for index in range(5)]
    results: list[int] = []
    start = threading.Barrier(len(browsers))

    def sign_in(browser: Browser) -> None:
        start.wait()
        results.append(browser.guest().status_code)

    threads = [threading.Thread(target=sign_in, args=(browser,)) for browser in browsers]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert sorted(results).count(200) == 1
    assert sorted(results).count(409) == 4
    assert one(db, "SELECT count(*) FROM active_session") == (1,)
    assert one(db, "SELECT count(*) FROM sessions") == (1,)


def test_a_seated_tab_can_read_and_keep_its_lease(guest: Browser) -> None:
    assert guest.get("/v1/queue").status_code == 200
    reply = guest.heartbeat()
    assert reply.status_code == 200
    assert reply.json()["lease_seconds"] == 45


def test_a_second_tab_is_detected_by_the_server(guest: Browser) -> None:
    guest.tab = TAB_B
    response = guest.get("/v1/queue")
    assert response.status_code == 409
    assert response.json()["code"] == "other_tab"


def test_a_duplicated_tab_counts_as_a_second_tab(guest: Browser) -> None:
    """Duplicate Tab copies the cookie, but a page load makes a new in-memory tab ID (F-036)."""
    cookie = guest.cookie()
    assert cookie
    guest.tab = "tab-duplicate"
    assert guest.get("/v1/feed").json()["code"] == "other_tab"


def test_the_second_tab_can_take_over_and_the_first_loses_the_seat(guest: Browser) -> None:
    first = guest.tab
    guest.tab = TAB_B
    refused = guest.post("/v1/gate/claim", json={"tab_id": TAB_B})
    assert refused.json()["code"] == "other_tab"
    assert (
        guest.post("/v1/gate/claim", json={"tab_id": TAB_B, "take_over": True}).status_code == 204
    )
    assert guest.get("/v1/queue").status_code == 200
    guest.tab = first
    assert guest.get("/v1/queue").json()["code"] == "other_tab"


def test_a_reload_releases_then_claims_with_a_new_tab(guest: Browser) -> None:
    released = guest.post("/v1/gate/release", json={"tab_id": guest.tab, "csrf_token": guest.csrf})
    assert released.status_code == 204
    guest.tab = "tab-reloaded"
    assert guest.post("/v1/gate/claim", json={"tab_id": guest.tab}).status_code == 204
    assert guest.get("/v1/queue").status_code == 200


def test_release_needs_the_token_in_its_body(guest: Browser) -> None:
    response = guest.post("/v1/gate/release", json={"tab_id": guest.tab, "csrf_token": "wrong"})
    assert response.status_code == 403


def test_a_crashed_tab_frees_the_seat_when_its_lease_expires(
    stack: Stack, guest: Browser, db: psycopg.Connection[Any]
) -> None:
    other = stack.browser(TAB_B)
    assert other.guest().status_code == 409
    db.execute("UPDATE active_session SET lease_expires_at = now() - interval '1 second'")
    db.commit()
    assert other.guest().status_code == 200
    assert guest.get("/v1/queue").json()["code"] == "seat_lost"
    assert guest.heartbeat().json()["code"] == "seat_lost"


def test_a_crashed_tab_can_come_back_if_nobody_took_the_seat(
    guest: Browser, db: psycopg.Connection[Any]
) -> None:
    db.execute("UPDATE active_session SET lease_expires_at = now() - interval '1 second'")
    db.commit()
    assert guest.get("/v1/queue").json()["code"] == "seat_lost"
    assert guest.post("/v1/gate/claim", json={"tab_id": guest.tab}).status_code == 204
    assert guest.get("/v1/queue").status_code == 200


def test_an_idle_guest_is_signed_out(guest: Browser, db: psycopg.Connection[Any]) -> None:
    db.execute("UPDATE sessions SET last_activity_at = now() - interval '6 minutes'")
    db.commit()
    response = guest.get("/v1/queue")
    assert response.status_code == 401
    assert response.json()["code"] == "session_ended"
    assert one(db, "SELECT count(*) FROM active_session") == (0,)
    assert one(db, "SELECT ended_reason FROM sessions") == ("expired",)


def test_heartbeats_alone_dont_keep_a_guest_active(
    guest: Browser, db: psycopg.Connection[Any]
) -> None:
    db.execute("UPDATE sessions SET last_activity_at = now() - interval '4 minutes'")
    db.commit()
    assert guest.heartbeat(active=False).status_code == 200
    before = one(db, "SELECT now() - last_activity_at > interval '3 minutes' FROM sessions")
    assert before == (True,)
    assert guest.heartbeat(active=True).status_code == 200
    after = one(db, "SELECT now() - last_activity_at < interval '1 minute' FROM sessions")
    assert after == (True,)


def test_activity_cant_extend_a_guest_past_the_total_limit(
    guest: Browser, db: psycopg.Connection[Any]
) -> None:
    db.execute("UPDATE sessions SET created_at = now() - interval '16 minutes'")
    db.commit()
    assert guest.heartbeat(active=True).status_code == 401


def test_the_guest_timer_counts_down(guest: Browser) -> None:
    left = guest.heartbeat().json()["guest_seconds_left"]
    assert 0 < left <= 300


def test_a_guest_must_wait_before_signing_in_again(stack: Stack, guest: Browser) -> None:
    assert guest.post("/v1/auth/logout").status_code == 204
    response = stack.browser(TAB_B).guest()
    assert response.status_code == 429
    assert response.json()["code"] == "guest_cooldown"


def test_the_wait_ends_after_a_minute(
    stack: Stack, guest: Browser, db: psycopg.Connection[Any]
) -> None:
    guest.post("/v1/auth/logout")
    db.execute("UPDATE sessions SET ended_at = now() - interval '61 seconds'")
    db.commit()
    assert stack.browser(TAB_B).guest().status_code == 200


def abandon(db: psycopg.Connection[Any], idle_for: str) -> None:
    """A guest who closed the tab: no sign-out, no release, and the lease long gone."""
    db.execute(
        "UPDATE sessions SET created_at = now() - %(ago)s::interval, "
        "last_activity_at = now() - %(ago)s::interval",
        {"ago": idle_for},
    )
    db.execute("UPDATE active_session SET lease_expires_at = now() - interval '1 second'")
    db.commit()


def test_an_abandoned_guest_session_doesnt_hold_off_the_next_guest(
    stack: Stack, guest: Browser, db: psycopg.Connection[Any]
) -> None:
    """Found in Phase 4 (F-109): the sweep dated the end now(), so the next guest from that
    address was always told to wait, and the refusal rolled the sweep back, so it never ended."""
    abandon(db, "10 minutes")
    assert stack.browser(TAB_B).guest().status_code == 200
    # Dated when the 5 idle minutes ran out, not when the sweep noticed.
    ended = one(
        db,
        "SELECT ended_at BETWEEN now() - interval '5 minutes 30 seconds' "
        "AND now() - interval '4 minutes 30 seconds' FROM sessions WHERE ended_reason = 'expired'",
    )
    assert ended == (True,)


def test_a_guest_who_ran_out_moments_ago_still_waits(
    stack: Stack, guest: Browser, db: psycopg.Connection[Any]
) -> None:
    abandon(db, "5 minutes 20 seconds")
    response = stack.browser(TAB_B).guest()
    assert response.status_code == 429
    assert response.json()["code"] == "guest_cooldown"
    assert 30 <= int(response.headers["retry-after"]) <= 41


def test_user_must_confirm_before_taking_the_seat_from_a_guest(
    stack: Stack, guest: Browser
) -> None:
    owner = stack.browser(TAB_B)
    response = owner.login()
    assert response.status_code == 409
    assert response.json()["code"] == "confirm_take_over"
    assert response.json()["holder_role"] == "guest"
    assert guest.get("/v1/queue").status_code == 200


def test_user_takes_the_seat_and_the_guest_is_out_immediately(
    stack: Stack, guest: Browser, db: psycopg.Connection[Any]
) -> None:
    owner = stack.browser(TAB_B)
    assert owner.login(take_over=True).status_code == 200
    assert owner.get("/v1/queue").status_code == 200
    response = guest.get("/v1/queue")
    assert response.status_code == 401
    assert response.json()["reason"] == "taken_over"
    assert one(db, "SELECT count(*) FROM audit_log WHERE action = 'gate.take_over'") == (1,)


def test_a_guest_can_never_displace_user(
    stack: Stack, user: Browser, db: psycopg.Connection[Any]
) -> None:
    visitor = stack.browser(TAB_B)
    assert visitor.guest().status_code == 409
    assert user.get("/v1/queue").status_code == 200


def test_a_signed_in_guest_cant_claim_a_seat_user_holds(
    stack: Stack, guest: Browser, db: psycopg.Connection[Any]
) -> None:
    """A guest whose tab crashed still has a session, and asks to take the seat by force."""
    db.execute("UPDATE active_session SET lease_expires_at = now() - interval '1 second'")
    db.commit()
    owner = stack.browser(TAB_B)
    assert owner.login().status_code == 200
    refused = guest.post("/v1/gate/claim", json={"tab_id": guest.tab, "take_over": True})
    assert refused.status_code == 409
    assert refused.json()["code"] == "seat_taken"
    assert owner.get("/v1/queue").status_code == 200


def test_user_can_take_the_seat_from_its_own_other_device(stack: Stack, user: Browser) -> None:
    laptop = stack.browser(TAB_B)
    assert laptop.login().json()["code"] == "confirm_take_over"
    assert laptop.login(take_over=True).status_code == 200
    assert user.get("/v1/queue").status_code == 401


def test_gate_status_is_public_and_says_nothing_about_who(stack: Stack, guest: Browser) -> None:
    visitor = stack.browser(TAB_B)
    body = visitor.client.get("/v1/gate/status").json()
    assert body["state"] == "in_use"
    assert body["mine"] is False
    assert set(body) == {"enabled", "state", "retry_after_seconds", "mine"}
    assert guest.get("/v1/gate/status").json()["mine"] is True


@pytest.mark.parametrize("stack", [{"gate_enabled": False}], indirect=True)
def test_with_the_gate_off_two_people_can_be_signed_in(stack: Stack) -> None:
    first, second = stack.browser(), stack.browser(TAB_B)
    assert first.guest().status_code == 200
    assert second.guest().status_code == 200
    assert first.get("/v1/queue").status_code == 200
    assert second.get("/v1/queue").status_code == 200
    assert first.heartbeat().status_code == 200
    assert second.get("/v1/gate/status").json()["enabled"] is False


@pytest.mark.parametrize("stack", [{"gate_enabled": False}], indirect=True)
def test_with_the_gate_off_roles_still_hold(stack: Stack) -> None:
    visitor = stack.browser()
    visitor.guest()
    assert visitor.post("/v1/admin/rollback", json={"reason": "a reason"}).status_code == 403
    nobody = stack.browser(TAB_B)
    assert nobody.get("/v1/queue").status_code == 401


def test_a_protected_request_needs_a_tab_id(guest: Browser) -> None:
    response = guest.client.get("/v1/queue")
    assert response.status_code == 400
    assert response.json()["code"] == "tab_required"


# --- The queue and feed ---


def test_queue_is_riskiest_first_and_pages_without_repeats(
    guest: Browser, db: psycopg.Connection[Any]
) -> None:
    for rev_id in range(1, 8):
        seed_edit(db, rev_id, score=rev_id / 10)
    seen: list[int] = []
    cursor = None
    pages = 0
    while True:
        params: dict[str, Any] = {"limit": 3}
        if cursor:
            params["cursor"] = cursor
        body = guest.get("/v1/queue", params=params).json()
        seen += [item["rev_id"] for item in body["items"]]
        pages += 1
        cursor = body["next_cursor"]
        if cursor is None:
            break
    assert seen == [7, 6, 5, 4, 3, 2, 1]
    assert pages == 3


def test_equal_scores_page_without_repeats_or_gaps(
    guest: Browser, db: psycopg.Connection[Any]
) -> None:
    for rev_id in range(1, 6):
        seed_edit(db, rev_id, score=0.5)
    first = guest.get("/v1/queue", params={"limit": 2}).json()
    second = guest.get("/v1/queue", params={"limit": 3, "cursor": first["next_cursor"]}).json()
    seen = [item["rev_id"] for item in first["items"] + second["items"]]
    assert seen == [5, 4, 3, 2, 1]


def test_the_queue_filters_by_editor_on_the_server(
    guest: Browser, db: psycopg.Connection[Any]
) -> None:
    seed_edit(db, 1, score=0.9, is_anonymous=True)
    seed_edit(db, 2, score=0.5, is_anonymous=False)

    def ids(editor: str) -> list[int]:
        body = guest.get("/v1/queue", params={"editor": editor}).json()
        return [item["rev_id"] for item in body["items"]]

    assert ids("all") == [1, 2]
    assert ids("logged_out") == [1]
    assert ids("registered") == [2]


def test_the_queue_returns_is_anonymous_for_the_label(
    guest: Browser, db: psycopg.Connection[Any]
) -> None:
    seed_edit(db, 1, is_anonymous=True)
    assert guest.get("/v1/queue").json()["items"][0]["is_anonymous"] is True


def test_editor_type_never_changes_the_order(guest: Browser, db: psycopg.Connection[Any]) -> None:
    """DEC-053: the model's score is the only ordering."""
    seed_edit(db, 1, score=0.3, is_anonymous=True)
    seed_edit(db, 2, score=0.6, is_anonymous=False)
    seed_edit(db, 3, score=0.9, is_anonymous=True)
    items = guest.get("/v1/queue").json()["items"]
    assert [item["rev_id"] for item in items] == [3, 2, 1]


def test_a_bad_editor_filter_is_refused(guest: Browser) -> None:
    assert guest.get("/v1/queue", params={"editor": "everyone"}).status_code == 422


def test_the_queue_only_goes_back_three_days(guest: Browser, db: psycopg.Connection[Any]) -> None:
    seed_edit(db, 1, edited_at=datetime.now(UTC) - timedelta(days=4))
    seed_edit(db, 2)
    assert [item["rev_id"] for item in guest.get("/v1/queue").json()["items"]] == [2]


def test_live_lists_and_health_say_they_are_live(
    guest: Browser, db: psycopg.Connection[Any]
) -> None:
    seed_edit(db, 1)
    assert guest.get("/v1/queue").json()["data_mode"] == "live"
    assert guest.get("/v1/feed").json()["data_mode"] == "live"
    health = guest.get("/v1/health/summary").json()
    assert (health["data_mode"], health["data_from"], health["data_to"]) == ("live", None, None)


SNAPSHOT = pytest.mark.parametrize("stack", [{"data_mode": "snapshot"}], indirect=True)


@SNAPSHOT
def test_a_snapshot_shows_every_edit_however_old(
    guest: Browser, db: psycopg.Connection[Any]
) -> None:
    """DEC-091: the demo's dataset is weeks old, and has no time window."""
    old = datetime(2026, 10, 27, 12, tzinfo=UTC)
    seed_edit(db, 1, score=0.9, edited_at=old)
    seed_edit(db, 2, score=0.4, edited_at=old + timedelta(days=5))
    seed_edit(db, 3, score=None, edited_at=old + timedelta(days=6))
    queue = guest.get("/v1/queue").json()
    assert [item["rev_id"] for item in queue["items"]] == [1, 2]
    assert queue["data_mode"] == "snapshot"
    feed = guest.get("/v1/feed", params={"limit": 2}).json()
    rest = guest.get("/v1/feed", params={"limit": 2, "cursor": feed["next_cursor"]}).json()
    assert [item["rev_id"] for item in feed["items"] + rest["items"]] == [3, 2, 1]
    assert feed["data_mode"] == "snapshot"


@SNAPSHOT
def test_a_snapshots_health_counts_the_whole_dataset(
    guest: Browser, db: psycopg.Connection[Any]
) -> None:
    first = datetime(2026, 10, 27, 12, tzinfo=UTC)
    last = first + timedelta(days=6)
    seed_edit(db, 1, edited_at=first)
    seed_edit(db, 2, edited_at=last)
    seed_edit(db, 3, score=None, edited_at=first + timedelta(days=1))
    # Scored long ago, so a live summary would count none of it.
    db.execute("UPDATE scores SET scored_at = %s", (first,))
    db.commit()
    body = guest.get("/v1/health/summary").json()
    assert body["data_mode"] == "snapshot"
    assert body["scored_last_24h"] == 2
    assert body["unscored_edits"] == 1
    assert datetime.fromisoformat(body["data_from"]) == first
    assert datetime.fromisoformat(body["data_to"]) == last


def test_unscored_edits_arent_in_the_queue_but_are_in_the_feed(
    guest: Browser, db: psycopg.Connection[Any]
) -> None:
    seed_edit(db, 1, score=None)
    assert guest.get("/v1/queue").json()["items"] == []
    feed = guest.get("/v1/feed").json()["items"]
    assert feed[0]["rev_id"] == 1
    assert feed[0]["score"] is None


def test_the_feed_is_newest_first_and_pages(guest: Browser, db: psycopg.Connection[Any]) -> None:
    now = datetime.now(UTC)
    for rev_id in range(1, 6):
        seed_edit(db, rev_id, edited_at=now - timedelta(minutes=rev_id))
    first = guest.get("/v1/feed", params={"limit": 2}).json()
    second = guest.get("/v1/feed", params={"limit": 3, "cursor": first["next_cursor"]}).json()
    assert [item["rev_id"] for item in first["items"] + second["items"]] == [1, 2, 3, 4, 5]


def test_page_size_has_a_maximum(guest: Browser) -> None:
    assert guest.get("/v1/queue", params={"limit": MAX_PAGE_SIZE + 1}).status_code == 422


@pytest.mark.parametrize(
    "cursor", ["garbage", "WyJhIl0", "e30", "' OR 1=1 --"], ids=["junk", "short", "object", "sql"]
)
def test_a_bad_cursor_is_refused(guest: Browser, cursor: str) -> None:
    response = guest.get("/v1/queue", params={"cursor": cursor})
    assert response.status_code == 400
    assert response.json()["code"] == "bad_cursor"


def test_lists_never_send_diff_text(guest: Browser, db: psycopg.Connection[Any]) -> None:
    seed_edit(db, 1)
    body = guest.get("/v1/queue").text + guest.get("/v1/feed").text
    assert "added words" not in body
    assert "removed words" not in body


def test_no_response_contains_the_editor_hash(guest: Browser, db: psycopg.Connection[Any]) -> None:
    seed_edit(db, 1)
    for path in ("/v1/queue", "/v1/feed", "/v1/edits/1"):
        assert "SECRET-HASH" not in guest.get(path).text
        assert "user_hash" not in guest.get(path).text


def test_a_poll_with_nothing_new_gets_304(guest: Browser, db: psycopg.Connection[Any]) -> None:
    seed_edit(db, 1)
    first = guest.get("/v1/queue")
    assert "private" in first.headers["cache-control"]
    second = guest.get("/v1/queue", headers={"if-none-match": first.headers["etag"]})
    assert second.status_code == 304
    assert second.content == b""


def test_large_lists_are_compressed(guest: Browser, db: psycopg.Connection[Any]) -> None:
    for rev_id in range(1, 40):
        seed_edit(db, rev_id)
    response = guest.client.get(
        "/v1/queue",
        headers={"x-tab-id": guest.tab, "accept-encoding": "gzip"},
        extensions={},
    )
    assert response.headers.get("content-encoding") == "gzip"
    assert gzip.compress(b"x")  # The stack handled decoding, and the body still parses.
    assert len(response.json()["items"]) == 39


# --- Edit detail ---


def test_edit_detail_has_text_scores_and_label(guest: Browser, db: psycopg.Connection[Any]) -> None:
    seed_edit(db, 1)
    db.execute("INSERT INTO labels (rev_id, label) VALUES (1, 1)")
    db.commit()
    body = guest.get("/v1/edits/1").json()
    assert body["added_text"] == "added words"
    assert body["scores"][0]["model_name"] == "roberta"
    assert body["label"] == 1
    assert body["wikipedia_url"] == "https://en.wikipedia.org/w/index.php?diff=1"
    assert body["has_explanation"] is False


def test_unknown_edit_is_404(guest: Browser) -> None:
    response = guest.get("/v1/edits/999")
    assert response.status_code == 404
    assert response.headers["content-type"] == "application/problem+json"


def test_hidden_edit_has_no_text(guest: Browser, db: psycopg.Connection[Any]) -> None:
    seed_edit(db, 1, hidden=True)
    body = guest.get("/v1/edits/1").json()
    assert body["content_hidden"] is True
    assert body["added_text"] is None


# --- Reviews ---


def test_user_can_review_and_edit_a_review(
    stack: Stack, user: Browser, db: psycopg.Connection[Any]
) -> None:
    seed_edit(db, 1)
    first = user.put("/v1/edits/1/review", json={"decision": "vandalism"})
    assert first.status_code == 200
    assert first.json()["updated"] is False
    second = user.put("/v1/edits/1/review", json={"decision": "ok", "note": "a note"})
    assert second.json()["updated"] is True
    assert one(db, "SELECT decision, note FROM reviews") == ("ok", "a note")
    assert one(db, "SELECT count(*) FROM audit_log WHERE action = 'review.saved'") == (2,)


@pytest.mark.parametrize("decision", ["vandalism", "good_faith_mistake", "ok", "skip"])
def test_all_four_decisions_are_accepted(
    user: Browser, db: psycopg.Connection[Any], decision: str
) -> None:
    seed_edit(db, 1)
    assert user.put("/v1/edits/1/review", json={"decision": decision}).status_code == 200


def test_a_reviewed_edit_leaves_the_queue_immediately(
    user: Browser, db: psycopg.Connection[Any]
) -> None:
    seed_edit(db, 1)
    assert len(user.get("/v1/queue").json()["items"]) == 1
    user.put("/v1/edits/1/review", json={"decision": "ok"})
    assert user.get("/v1/queue").json()["items"] == []
    shown = user.get("/v1/queue", params={"include_reviewed": True}).json()["items"]
    assert shown[0]["decision"] == "ok"


def test_reviewing_an_unknown_edit_is_404(user: Browser) -> None:
    assert user.put("/v1/edits/999/review", json={"decision": "ok"}).status_code == 404


def test_an_unknown_field_is_refused(user: Browser, db: psycopg.Connection[Any]) -> None:
    seed_edit(db, 1)
    response = user.put("/v1/edits/1/review", json={"decision": "ok", "role": "admin"})
    assert response.status_code == 422


# --- Models and health ---


def test_the_leaderboard_lists_the_champion_first(
    guest: Browser, db: psycopg.Connection[Any]
) -> None:
    for name, version, alias in (("mlp", "5", "fallback-4"), ("roberta", "1", "champion")):
        db.execute(
            "INSERT INTO model_versions (model_name, model_version, alias, feature_version, "
            "snapshot_hash, artifact_sha256, metrics) VALUES (%s, %s, %s, 1, 'snap', 'sha', %s)",
            (name, version, alias, json.dumps({"pr_auc": 0.3})),
        )
    db.commit()
    items = guest.get("/v1/models").json()["items"]
    assert [item["model_name"] for item in items] == ["roberta", "mlp"]
    assert items[0]["metrics"] == {"pr_auc": 0.3}


def test_the_health_summary_counts_fallbacks(
    stack: Stack, guest: Browser, db: psycopg.Connection[Any]
) -> None:
    seed_edit(db, 1)
    seed_edit(db, 2)
    seed_edit(db, 3, score=None)
    db.execute("UPDATE scores SET used_fallback = true WHERE rev_id = 1")
    db.commit()
    body = guest.get("/v1/health/summary").json()
    assert body["scored_last_24h"] == 2
    assert body["fallback_rate_24h"] == 0.5
    assert body["unscored_edits"] == 1
    assert body["models_loaded"] is True
    assert [item["breaker"] for item in body["chain"]] == ["closed", "closed"]


def test_the_health_summary_carries_each_retrain_report(
    guest: Browser, db: psycopg.Connection[Any]
) -> None:
    """The Health page shows the trainer's before vs after report from details (DEC-066)."""
    report = {"before": {"model": "roberta", "pr_auc": 0.36}, "after": {"pr_auc": 0.37}}
    db.execute(
        "INSERT INTO retrain_runs (trigger, status, details) VALUES ('scheduled', 'done', %s)",
        (Jsonb(report),),
    )
    db.commit()
    [run] = guest.get("/v1/health/summary").json()["retrain_runs"]
    assert run["trigger"] == "scheduled"
    assert run["details"] == report
    assert isinstance(run["id"], int)


def test_the_trainers_bookkeeping_never_leaves_the_database(
    guest: Browser, db: psycopg.Connection[Any]
) -> None:
    """Listings, slice tables, and alias maps stay behind, to spare Supabase's egress."""
    details = {
        "before": {"pr_auc": 0.36},
        "reasons": ["drift"],
        "gate": {"passed": False, "checks": []},
        "listings": {"6": {"metrics": {"card": "large"}}},
        "slices": {"anonymous": {}},
        "aliases_before": {"champion": "1"},
        "proposed_aliases": {"champion": "6"},
    }
    db.execute(
        "INSERT INTO retrain_runs (trigger, status, details) VALUES ('drift', 'shadow', %s)",
        (Jsonb(details),),
    )
    db.commit()
    [run] = guest.get("/v1/health/summary").json()["retrain_runs"]
    assert set(run["details"]) == {"before", "reasons", "gate"}


# --- /predict ---


def test_predict_scores_text_and_says_which_model(stack: Stack, guest: Browser) -> None:
    response = guest.post("/v1/predict", json={"added_text": "some words"})
    assert response.status_code == 200
    body = response.json()
    assert body["score"] == 0.8
    assert body["model_name"] == "roberta"
    assert body["used_fallback"] is False
    assert body["source"] == "model"


def test_predict_on_a_failing_champion_falls_back_and_flags_it(
    stack: Stack, guest: Browser
) -> None:
    stack.models["roberta"].fail = True
    body = guest.post("/v1/predict", json={"added_text": "other words"}).json()
    assert body["model_name"] == "mlp"
    assert body["used_fallback"] is True


def test_predict_returns_503_when_every_model_fails(stack: Stack, guest: Browser) -> None:
    for model in stack.models.values():
        model.fail = True
    response = guest.post("/v1/predict", json={"added_text": "x"})
    assert response.status_code == 503
    assert response.json()["code"] == "models_unavailable"


def test_repeating_the_same_text_doesnt_repeat_inference(stack: Stack, user: Browser) -> None:
    body = {"added_text": "repeat me"}
    assert user.post("/v1/predict", json=body).json()["source"] == "model"
    assert user.post("/v1/predict", json=body).json()["source"] == "cache"
    assert stack.models["roberta"].calls == 1


def test_a_known_edit_reads_its_stored_score(
    stack: Stack, guest: Browser, db: psycopg.Connection[Any]
) -> None:
    seed_edit(db, 1, score=0.4)
    body = guest.post("/v1/predict", json={"rev_id": 1}).json()
    assert body["source"] == "stored"
    assert body["score"] == pytest.approx(0.4)
    assert stack.models["roberta"].calls == 0


def test_predict_on_an_unknown_edit_is_404(guest: Browser) -> None:
    assert guest.post("/v1/predict", json={"rev_id": 999}).status_code == 404


@pytest.mark.parametrize(
    "body",
    [
        {"added_text": "x" * 20_001},
        {"comment": "x" * 1001},
        {"old_length": -1},
        {"user_edit_count": -1},
        {"surprise": True},
    ],
    ids=["long-text", "long-comment", "negative-length", "negative-count", "unknown-field"],
)
def test_predict_input_limits(guest: Browser, body: dict[str, Any]) -> None:
    assert guest.post("/v1/predict", json=body).status_code == 422


def test_a_guest_has_a_stricter_predict_limit(guest: Browser) -> None:
    statuses = [
        guest.post("/v1/predict", json={"added_text": f"text {index}"}).status_code
        for index in range(11)
    ]
    assert statuses[:10] == [200] * 10
    assert statuses[10] == 429


def test_user_has_a_higher_predict_limit(user: Browser) -> None:
    statuses = [
        user.post("/v1/predict", json={"added_text": f"text {index}"}).status_code
        for index in range(11)
    ]
    assert 429 not in statuses


def test_a_rate_limit_answer_says_when_to_retry(guest: Browser) -> None:
    for index in range(10):
        guest.post("/v1/predict", json={"added_text": f"text {index}"})
    response = guest.post("/v1/predict", json={"added_text": "one more"})
    assert response.status_code == 429
    assert int(response.headers["retry-after"]) > 0
    assert response.json()["code"] == "rate_limited"


def test_the_concurrency_cap_returns_503_without_queueing(stack: Stack, user: Browser) -> None:
    slots = stack.app.state.context.predict_slots
    for _ in range(stack.settings.predict_max_concurrency):
        slots.acquire()
    response = user.post("/v1/predict", json={"added_text": "while busy"})
    for _ in range(stack.settings.predict_max_concurrency):
        slots.release()
    assert response.status_code == 503
    assert response.json()["code"] == "predict_busy"
    assert response.headers["retry-after"] == "2"


@pytest.mark.parametrize("stack", [{"force_model": "mlp"}], indirect=True)
def test_force_model_is_visible_in_the_answer(stack: Stack, guest: Browser) -> None:
    body = guest.post("/v1/predict", json={"added_text": "x"}).json()
    assert body["model_name"] == "mlp"
    assert body["forced"] is True


# --- Startup ---


def test_ready_is_true_once_models_and_database_are_up(stack: Stack) -> None:
    response = stack.browser().client.get("/ready")
    assert response.status_code == 200
    assert response.json() == {"ready": True, "model_loaded": True, "database": True}


def test_ready_is_false_until_the_model_loads_but_reads_still_work(
    db: psycopg.Connection[Any], api_url: str, tmp_path: Any
) -> None:
    built = Stack(api_url, tmp_path)
    seed_edit(db, 1)
    built.app.state.context.db.open()  # The database is up, and models never got loaded.
    browser = built.browser()
    assert browser.client.get("/health").status_code == 200
    response = browser.client.get("/ready")
    assert response.status_code == 503
    assert response.json()["model_loaded"] is False
    browser.guest()
    assert len(browser.get("/v1/queue").json()["items"]) == 1
    busy = browser.post("/v1/predict", json={"added_text": "x"})
    assert busy.status_code == 503
    assert busy.json()["code"] == "model_loading"
    assert int(busy.headers["retry-after"]) > 0
    built.app.state.context.db.close()


@pytest.mark.parametrize("stack", [{"predict_enabled": False}], indirect=True)
def test_ready_doesnt_wait_for_a_model_with_predict_off(stack: Stack) -> None:
    assert stack.browser().client.get("/ready").json()["ready"] is True


def test_a_failed_model_load_keeps_the_app_up_and_not_ready(
    db: psycopg.Connection[Any], api_url: str, tmp_path: Any
) -> None:
    from fastapi.testclient import TestClient

    built = Stack(api_url, tmp_path, force_model="not-a-model")
    with TestClient(built.app) as client:
        assert client.get("/health").status_code == 200
        deadline = datetime.now(UTC) + timedelta(seconds=5)
        while built.app.state.context.model_error is None and datetime.now(UTC) < deadline:
            pass
        assert client.get("/ready").status_code == 503


# --- Feature flags ---


@pytest.mark.parametrize("stack", [{"read_only_mode": True}], indirect=True)
def test_read_only_mode_blocks_writes_and_allows_reads(
    stack: Stack, db: psycopg.Connection[Any]
) -> None:
    seed_edit(db, 1)
    owner = stack.browser()
    owner.login()
    assert owner.get("/v1/queue").status_code == 200
    review = owner.put("/v1/edits/1/review", json={"decision": "ok"})
    assert review.status_code == 503
    assert review.json()["code"] == "read_only"
    assert owner.post("/v1/admin/retrain", json={"reason": "a reason"}).status_code == 503


@pytest.mark.parametrize("stack", [{"predict_enabled": False}], indirect=True)
def test_predict_can_be_switched_off(stack: Stack) -> None:
    browser = stack.browser()
    browser.guest()
    response = browser.post("/v1/predict", json={"added_text": "x"})
    assert response.status_code == 503
    assert response.json()["code"] == "feature_disabled"


@pytest.mark.parametrize("stack", [{"explain_enabled": False}], indirect=True)
def test_explain_can_be_switched_off(stack: Stack, db: psycopg.Connection[Any]) -> None:
    seed_edit(db, 1)
    browser = stack.browser()
    browser.guest()
    assert browser.post("/v1/explain/1").status_code == 503
    assert stack.provider.calls == []


# --- API foundations ---


def test_errors_use_one_format_with_a_request_id(guest: Browser) -> None:
    response = guest.get("/v1/edits/999", headers={"x-request-id": "my-request-1234"})
    body = response.json()
    assert response.headers["content-type"] == "application/problem+json"
    assert body["type"] == "/problems/edit_not_found"
    assert body["status"] == 404
    assert body["request_id"] == "my-request-1234"
    assert response.headers["x-request-id"] == "my-request-1234"


def test_a_hostile_request_id_is_replaced(guest: Browser) -> None:
    response = guest.get("/v1/queue", headers={"x-request-id": "bad id\nwith newline"[:6]})
    assert response.headers["x-request-id"] != "bad id"


def test_an_unknown_path_is_a_problem_too(stack: Stack) -> None:
    response = stack.browser().client.get("/v1/nothing-here")
    assert response.status_code == 404
    assert response.headers["content-type"] == "application/problem+json"


def test_validation_errors_dont_echo_the_input(user: Browser) -> None:
    response = user.put("/v1/edits/1/review", json={"decision": "<script>alert(1)</script>"})
    assert response.status_code == 422
    assert "script" not in response.text


def test_unexpected_errors_show_nothing_internal(stack: Stack, user: Browser) -> None:
    def explode(*_: Any) -> Any:
        raise RuntimeError("password=hunter2 at SELECT secret FROM table")

    stack.app.state.context.access.seated = explode
    client = stack.browser().client
    response = client.get("/v1/queue", headers={"x-tab-id": user.tab})
    assert response.status_code == 500
    assert "hunter2" not in response.text
    assert "SELECT" not in response.text
    assert response.json()["code"] == "internal_error"
    assert response.headers["x-content-type-options"] == "nosniff"


def test_a_slow_query_is_cancelled(stack: Stack) -> None:
    database = stack.app.state.context.db
    database.statement_timeout_ms = 50
    with pytest.raises(DatabaseUnavailable), database.tx() as tx:
        tx.one("SELECT pg_sleep(2)")


def test_the_openapi_schema_has_public_routes_and_hides_admin(stack: Stack) -> None:
    schema = stack.browser().client.get("/v1/openapi.json").json()
    assert "/v1/queue" in schema["paths"]
    assert not [path for path in schema["paths"] if "admin" in path]


def test_the_schema_documents_errors_in_the_format_the_api_sends(stack: Stack) -> None:
    client = stack.browser().client
    schema = client.get("/v1/openapi.json").json()
    documented = schema["paths"]["/v1/auth/guest"]["post"]["responses"]["422"]["content"]
    assert list(documented) == ["application/problem+json"]
    assert "HTTPValidationError" not in schema["components"]["schemas"]
    sent = client.post("/v1/auth/guest", json={})
    assert sent.status_code == 422
    assert sent.headers["content-type"].startswith("application/problem+json")
    wanted = set(schema["components"]["schemas"]["ProblemDetails"]["required"])
    assert wanted <= set(sent.json())


def test_all_public_paths_are_versioned(stack: Stack) -> None:
    for method, path in all_routes(stack):
        if path not in {"/health", "/ready", "/metrics"}:
            assert path.startswith("/v1/"), (method, path)


# --- Hardening ---


def test_security_headers_are_on_every_response(stack: Stack) -> None:
    for path in ("/health", "/v1/nothing"):
        headers = stack.browser().client.get(path).headers
        assert headers["x-content-type-options"] == "nosniff"
        assert headers["referrer-policy"] == "no-referrer"
        assert headers["x-frame-options"] == "DENY"
        assert "frame-ancestors 'none'" in headers["content-security-policy"]
        assert "x-request-id" in headers
        assert "server" not in headers


def test_authenticated_answers_arent_cached_by_default(guest: Browser) -> None:
    assert guest.get("/v1/auth/me").headers["cache-control"] == "no-store"


def test_a_large_body_is_refused_by_its_declared_size(stack: Stack) -> None:
    response = stack.browser().client.post(
        "/v1/auth/guest",
        content=b"x",
        headers={"content-length": str(stack.settings.max_body_bytes + 1)},
    )
    assert response.status_code == 413


def test_a_large_chunked_body_is_refused(stack: Stack) -> None:
    def chunks() -> Any:
        for _ in range(20):
            yield b"x" * 10_000

    with stack.browser().client as client:
        response = client.post(
            "/v1/auth/guest",
            content=chunks(),
            headers={"content-type": "application/json"},
        )
    assert response.status_code in {400, 413, 422}


def test_metrics_need_the_bearer_token(stack: Stack) -> None:
    client = stack.browser().client
    assert client.get("/metrics").status_code == 401
    assert client.get("/metrics", headers={"authorization": "Bearer wrong"}).status_code == 401
    good = client.get("/metrics", headers={"authorization": f"Bearer {METRICS_TOKEN}"})
    assert good.status_code == 200
    assert "api_requests_total" in good.text


def test_metrics_label_routes_by_template_so_labels_stay_bounded(
    guest: Browser, stack: Stack, db: psycopg.Connection[Any]
) -> None:
    seed_edit(db, 1)
    guest.get("/v1/edits/1")
    guest.get("/v1/edits/2")
    guest.get("/v1/no/such/path/123")
    text = (
        stack.browser()
        .client.get("/metrics", headers={"authorization": f"Bearer {METRICS_TOKEN}"})
        .text
    )
    assert 'route="/v1/edits/{rev_id}"' in text
    assert "/v1/edits/1" not in text
    assert "no/such/path" not in text


def test_cors_allows_only_the_listed_origin(stack: Stack) -> None:
    client = stack.browser().client
    allowed = client.options(
        "/v1/queue",
        headers={"origin": "http://localhost:3000", "access-control-request-method": "GET"},
    )
    assert allowed.headers["access-control-allow-origin"] == "http://localhost:3000"
    assert allowed.headers["access-control-allow-credentials"] == "true"
    refused = client.options(
        "/v1/queue",
        headers={"origin": "https://evil.example", "access-control-request-method": "GET"},
    )
    assert "access-control-allow-origin" not in refused.headers


def test_the_general_read_limit_exists(guest: Browser) -> None:
    statuses = [guest.get("/v1/auth/me").status_code for _ in range(121)]
    assert statuses[-1] == 429


# --- Admin requests ---


def test_a_retrain_request_is_a_row_and_nothing_else(
    user: Browser, db: psycopg.Connection[Any]
) -> None:
    response = user.post("/v1/admin/retrain", json={"reason": "drift looks bad"})
    assert response.status_code == 202
    body = response.json()
    assert body["status"] == "pending"
    assert body["kind"] == "retrain"
    assert one(db, "SELECT kind, reason, requested_by FROM retrain_requests") == (
        "retrain",
        "drift looks bad",
        "user",
    )
    assert one(db, "SELECT count(*) FROM audit_log WHERE action = 'admin.retrain'") == (1,)


def test_repeating_a_request_doesnt_add_a_second(
    user: Browser, db: psycopg.Connection[Any]
) -> None:
    first = user.post("/v1/admin/retrain", json={"reason": "first"}).json()
    second = user.post("/v1/admin/retrain", json={"reason": "second"}).json()
    assert first["id"] == second["id"]
    assert one(db, "SELECT count(*) FROM retrain_requests") == (1,)


def test_promote_records_the_version(user: Browser, db: psycopg.Connection[Any]) -> None:
    user.post("/v1/admin/promote", json={"reason": "better model", "model_version": "7"})
    assert one(db, "SELECT kind, details FROM retrain_requests") == (
        "promote",
        {"model_version": "7"},
    )


def test_the_request_list_shows_history(user: Browser) -> None:
    user.post("/v1/admin/rollback", json={"reason": "bad model"})
    items = user.get("/v1/admin/requests").json()["items"]
    assert items[0]["kind"] == "rollback"


def test_a_reason_is_required(user: Browser) -> None:
    assert user.post("/v1/admin/retrain", json={"reason": ""}).status_code == 422


def test_the_audit_log_is_readable_and_pages(user: Browser, db: psycopg.Connection[Any]) -> None:
    for _ in range(3):
        db.execute("INSERT INTO audit_log (actor, action) VALUES ('t', 'test.event')")
    db.commit()
    first = user.get("/v1/admin/audit-log", params={"limit": 2}).json()
    assert len(first["items"]) == 2
    second = user.get("/v1/admin/audit-log", params={"cursor": first["next_cursor"]}).json()
    seen = {item["id"] for item in first["items"]} | {item["id"] for item in second["items"]}
    assert len(seen) == len(first["items"]) + len(second["items"])


# --- Postgres down (section 14) ---


def break_database(stack: Stack) -> None:
    stack.app.state.context.db.pool.close()


def test_with_postgres_down_stale_queue_is_served_and_marked(
    stack: Stack, guest: Browser, db: psycopg.Connection[Any]
) -> None:
    seed_edit(db, 1)
    assert guest.get("/v1/queue").json()["stale"] is False
    stack.app.state.context.queue_cache._entries[("all", 50, None, False)].expires_at = 0
    break_database(stack)
    response = guest.get("/v1/queue")
    assert response.status_code == 200
    assert response.json()["stale"] is True
    assert response.json()["items"][0]["rev_id"] == 1


def test_with_postgres_down_new_sign_ins_are_refused(stack: Stack) -> None:
    break_database(stack)
    response = stack.browser().guest()
    assert response.status_code == 503
    assert response.json()["code"] == "database_unavailable"


def test_with_postgres_down_a_session_lasts_until_its_lease_ends(
    stack: Stack, guest: Browser
) -> None:
    assert guest.get("/v1/auth/me").status_code == 200
    guest.get("/v1/queue")
    break_database(stack)
    context = stack.app.state.context
    principal: Principal = context.access.seated(guest.cookie(), guest.tab)
    assert principal.role == "guest"
    with context.access._recent_lock:
        key = next(iter(context.access._recent))
        known = context.access._recent[key]
        context.access._recent[key] = (known[0], 0.0)
    with pytest.raises(DatabaseUnavailable):
        context.access.seated(guest.cookie(), guest.tab)


def test_with_postgres_down_ready_is_false_and_health_is_true(stack: Stack) -> None:
    break_database(stack)
    client = stack.browser().client
    stack.app.state.context._ready_checked_at = 0
    assert client.get("/health").status_code == 200
    assert client.get("/ready").status_code == 503


def test_the_api_role_cant_touch_what_it_shouldnt(stack: Stack) -> None:
    """The API's own connection, so a compromised API is still limited by its role."""
    database = stack.app.state.context.db
    for statement in (
        "UPDATE labels SET label = 1",
        "UPDATE model_versions SET alias = 'champion'",
        "DELETE FROM audit_log",
        "DELETE FROM edits",
        "UPDATE edits SET title = 'x'",
    ):
        with pytest.raises(psycopg.errors.InsufficientPrivilege), database.tx() as tx:
            tx.run(statement)


def test_the_password_never_appears_in_logs_or_errors(
    stack: Stack, caplog: pytest.LogCaptureFixture
) -> None:
    browser = stack.browser()
    browser.login(PASSWORD)
    browser.login("wrong-secret-value")
    assert PASSWORD not in caplog.text
    assert "wrong-secret-value" not in caplog.text


# --- Caching ---


def test_the_queue_cache_saves_a_query_within_its_ttl(
    stack: Stack, guest: Browser, db: psycopg.Connection[Any]
) -> None:
    seed_edit(db, 1)
    guest.get("/v1/queue")
    stack.statements.clear()
    guest.get("/v1/queue")
    assert not [query for query in stack.statements if "e.latest_score DESC" in query]
    registry = stack.app.state.context.metrics.registry
    assert (
        registry.get_sample_value("api_cache_events_total", {"cache": "queue", "event": "hit"}) == 1
    )


def test_the_leaderboard_cache_clears_when_the_alias_changes(
    stack: Stack, guest: Browser, db: psycopg.Connection[Any]
) -> None:
    from wikiguard_core.chain import ModelRef

    db.execute(
        "INSERT INTO model_versions (model_name, model_version, alias, feature_version, "
        "snapshot_hash, artifact_sha256) VALUES ('roberta', '1', 'champion', 1, 's', 'a')"
    )
    db.commit()
    assert guest.get("/v1/models").json()["items"][0]["model_version"] == "1"
    db.execute("UPDATE model_versions SET model_version = '9'")
    db.commit()
    assert guest.get("/v1/models").json()["items"][0]["model_version"] == "1"

    source = stack.chain._source
    existing = source.refs[0]
    source.refs[0] = ModelRef(existing.alias, "roberta", "9", existing.artifact_sha256)
    assert stack.chain.reload() is True
    assert guest.get("/v1/models").json()["items"][0]["model_version"] == "9"


def test_the_predict_cache_clears_when_the_alias_changes(stack: Stack, user: Browser) -> None:
    from wikiguard_core.chain import ModelRef

    body = {"added_text": "same text"}
    user.post("/v1/predict", json=body)
    source = stack.chain._source
    existing = source.refs[0]
    source.refs[0] = ModelRef(existing.alias, "roberta", "9", existing.artifact_sha256)
    stack.chain.reload()
    reply = user.post("/v1/predict", json=body).json()
    assert reply["source"] == "model"
    assert reply["model_version"] == "9"


# --- No N+1 queries ---


def statements_for(stack: Stack, browser: Browser, path: str) -> int:
    context = stack.app.state.context
    for cache in (context.queue_cache, context.feed_cache):
        cache.clear()
    stack.statements.clear()
    assert browser.get(path).status_code == 200
    return len(stack.statements)


@pytest.mark.parametrize("path", ["/v1/queue", "/v1/feed"])
def test_query_count_doesnt_grow_with_the_page(
    stack: Stack, guest: Browser, db: psycopg.Connection[Any], path: str
) -> None:
    seed_edit(db, 1)
    with_one = statements_for(stack, guest, path)
    for rev_id in range(2, 51):
        seed_edit(db, rev_id)
    assert statements_for(stack, guest, path) == with_one


def test_edit_detail_query_count_doesnt_grow_with_its_scores(
    stack: Stack, guest: Browser, db: psycopg.Connection[Any]
) -> None:
    seed_edit(db, 1)
    with_one = statements_for(stack, guest, "/v1/edits/1")
    for version in range(2, 12):
        db.execute(
            "INSERT INTO scores (rev_id, model_name, model_version, score) "
            "VALUES (1, 'mlp', %s, 0.1)",
            (str(version),),
        )
    db.commit()
    assert statements_for(stack, guest, "/v1/edits/1") == with_one
