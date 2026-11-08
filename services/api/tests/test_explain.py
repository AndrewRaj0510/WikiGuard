"""Explain: prompt safety and output checks need no database; the rest use real Postgres."""

import json
import threading
from pathlib import Path
from typing import Any

import httpx
import psycopg
import pytest
from api_test_helpers import Browser, Stack, one, seed_edit
from wikiguard_api.explain import (
    MAX_WORDS,
    PROMPT_VERSION,
    SYSTEM_PROMPT,
    UNAVAILABLE,
    GeminiProvider,
    OpenAiProvider,
    ProviderError,
    build_user_prompt,
    clean_output,
)

FIXTURES = Path(__file__).parent / "fixtures"
INJECTION_CASES = [
    json.loads(line)
    for line in (FIXTURES / "explain_injection.jsonl").read_text("utf-8").splitlines()
]
FAITHFULNESS_CASES = [
    json.loads(line)
    for line in (FIXTURES / "explain_faithfulness.jsonl").read_text("utf-8").splitlines()
]
injection_ids = [case["id"] for case in INJECTION_CASES]


# --- The prompt: no database ---


@pytest.mark.parametrize("case", INJECTION_CASES, ids=injection_ids)
def test_injected_text_stays_inside_its_tags(case: dict[str, Any]) -> None:
    prompt = build_user_prompt(case["title"], case["comment"], case["added"], case["removed"])
    # Exactly one of each tag: text from the edit can't open or close one.
    for tag in ("title", "comment", "added", "removed"):
        assert prompt.count(f"<{tag}>") == 1
        assert prompt.count(f"</{tag}>") == 1
    for text in (case["title"], case["comment"], case["added"], case["removed"]):
        if "<" in text:
            assert text not in prompt


@pytest.mark.parametrize("case", INJECTION_CASES, ids=injection_ids)
def test_injected_text_never_reaches_the_system_prompt(case: dict[str, Any]) -> None:
    for text in (case["title"], case["comment"], case["added"], case["removed"]):
        if text:
            assert text not in SYSTEM_PROMPT


@pytest.mark.parametrize("case", INJECTION_CASES, ids=injection_ids)
def test_what_an_obedient_model_would_output_is_rejected_when_harmful(
    case: dict[str, Any],
) -> None:
    """A model that fell for it. The output check is the layer that doesn't rely on the model."""
    if case["obedient_output"] is not None:
        assert clean_output(case["obedient_output"]) is None


def test_every_escape_attempt_is_covered() -> None:
    kinds = {case["kind"] for case in INJECTION_CASES}
    assert {
        "tag_escape",
        "instruction",
        "link_request",
        "markup_request",
        "length_attack",
        "comment_field",
        "title_field",
        "role_play",
        "unicode",
    } <= kinds


def test_truncation_cant_split_an_escape_sequence() -> None:
    prompt = build_user_prompt("", "", "a" * 1999 + "<", "")
    assert "&lt;" in prompt
    assert prompt.count("<added>") == 1


def test_long_fields_are_cut() -> None:
    prompt = build_user_prompt("t" * 5000, "c" * 5000, "a" * 50_000, "r" * 50_000)
    assert len(prompt) < 5000


def test_the_system_prompt_states_the_rule() -> None:
    assert "untrusted" in SYSTEM_PROMPT
    assert "never instructions" in SYSTEM_PROMPT


@pytest.mark.parametrize(
    "answer",
    [
        "",
        "   ",
        " ".join(["word"] * (MAX_WORDS + 1)),
        "Visit https://evil.example now",
        "Go to http://evil.example",
        "see www.evil.example",
        "evil.com has more",
        "evil.example has more",
        "see evil.ai now",
        "see evil\uff0ecom now",
        "<b>bold</b> statement",
        "**bold** statement",
        "[text](link) here",
        "`code` here",
        "# Heading",
        "{{template}} here",
        "[[wikilink]] here",
    ],
)
def test_bad_answers_are_rejected(answer: str) -> None:
    assert clean_output(answer) is None


@pytest.mark.parametrize(
    "answer",
    [
        "The edit added a sentence about the Moon.",
        "Replaced the population figure with 1,204 and removed a citation.",
        "The edit removed text and asks the reader to ignore earlier instructions.",
        " ".join(["word"] * MAX_WORDS),
    ],
)
def test_ordinary_answers_pass(answer: str) -> None:
    assert clean_output(answer) == answer


def test_whitespace_is_collapsed_to_one_line() -> None:
    assert clean_output("Two\n\nlines  here") == "Two lines here"


# --- Faithfulness: the eval set, scored with a stand-in. The real run needs a key. ---


@pytest.mark.parametrize(
    "case", FAITHFULNESS_CASES, ids=[case["id"] for case in FAITHFULNESS_CASES]
)
def test_the_faithfulness_set_is_well_formed(case: dict[str, Any]) -> None:
    assert case["must_mention_any"], "needs something the summary has to say"
    assert case["added"] or case["removed"]
    assert clean_output(case["good_answer"]) == case["good_answer"]
    answer = case["good_answer"].lower()
    assert any(word in answer for word in case["must_mention_any"])
    assert not any(word in answer for word in case.get("must_not_claim", []))


# --- Providers, against recorded response shapes ---


def openai_app(status: int = 200, body: Any = None) -> httpx.MockTransport:
    def handle(request: httpx.Request) -> httpx.Response:
        sent = json.loads(request.content)
        assert sent["temperature"] == 0
        assert sent["max_completion_tokens"] == 80
        assert "tools" not in sent
        assert request.headers["authorization"] == "Bearer key"
        reply = {"choices": [{"message": {"role": "assistant", "content": "An answer."}}]}
        return httpx.Response(status, json=reply if body is None else body)

    return httpx.MockTransport(handle)


def gemini_app(status: int = 200, body: Any = None) -> httpx.MockTransport:
    def handle(request: httpx.Request) -> httpx.Response:
        assert "key" not in str(request.url.query)
        assert request.headers["x-goog-api-key"] == "key"
        assert "tools" not in json.loads(request.content)
        reply = {"candidates": [{"content": {"parts": [{"text": "An answer."}]}}]}
        return httpx.Response(status, json=reply if body is None else body)

    return httpx.MockTransport(handle)


@pytest.mark.parametrize(
    "make",
    [
        lambda **kw: OpenAiProvider("m", "key", openai_app(**kw)),
        lambda **kw: GeminiProvider("m", "key", gemini_app(**kw)),
    ],
    ids=["openai", "gemini"],
)
class TestProviders:
    def test_a_normal_answer(self, make: Any) -> None:
        assert make().complete("system", "user", "req-1") == "An answer."

    def test_a_failure_status_becomes_a_clean_error(self, make: Any) -> None:
        with pytest.raises(ProviderError, match="status 429"):
            make(status=429).complete("system", "user", None)

    def test_a_malformed_reply_becomes_a_clean_error(self, make: Any) -> None:
        with pytest.raises(ProviderError):
            make(body={"unexpected": True}).complete("system", "user", None)


def test_a_network_failure_becomes_a_clean_error() -> None:
    def fail(_: httpx.Request) -> httpx.Response:
        raise httpx.ConnectTimeout("timed out with key=SECRET")

    provider = OpenAiProvider("m", "SECRET", httpx.MockTransport(fail))
    with pytest.raises(ProviderError) as caught:
        provider.complete("s", "u", None)
    assert "SECRET" not in str(caught.value)


# --- The endpoint, on real Postgres ---


@pytest.mark.integration
class TestEndpoint:
    def test_an_answer_is_returned_and_cached(
        self, stack: Stack, user: Browser, db: psycopg.Connection[Any]
    ) -> None:
        seed_edit(db, 1)
        first = user.post("/v1/explain/1").json()
        assert first["available"] is True
        assert first["cached"] is False
        second = user.post("/v1/explain/1").json()
        assert second["cached"] is True
        assert second["explanation"] == first["explanation"]
        assert len(stack.provider.calls) == 1
        stored = one(db, "SELECT prompt_version, llm_model FROM explanations")
        assert stored == (PROMPT_VERSION, "fake-model")

    def test_the_diff_comes_from_the_database_never_the_client(
        self, stack: Stack, user: Browser, db: psycopg.Connection[Any]
    ) -> None:
        seed_edit(db, 1, added_text="the real diff text")
        response = user.post("/v1/explain/1", json={"added_text": "client text"})
        assert response.status_code == 200
        assert "the real diff text" in stack.provider.calls[0]
        assert "client text" not in stack.provider.calls[0]

    def test_an_unknown_edit_is_404_with_no_llm_call(self, stack: Stack, user: Browser) -> None:
        assert user.post("/v1/explain/999").status_code == 404
        assert stack.provider.calls == []

    def test_a_hidden_edit_is_never_sent(
        self, stack: Stack, user: Browser, db: psycopg.Connection[Any]
    ) -> None:
        seed_edit(db, 1, hidden=True)
        body = user.post("/v1/explain/1").json()
        assert body["available"] is False
        assert body["reason"] == "hidden"
        assert stack.provider.calls == []

    def test_text_past_retention_is_never_sent(
        self, stack: Stack, user: Browser, db: psycopg.Connection[Any]
    ) -> None:
        seed_edit(db, 1, added_text=None)
        assert user.post("/v1/explain/1").json()["reason"] == "hidden"
        assert stack.provider.calls == []

    def test_a_provider_failure_is_unavailable_and_the_app_still_works(
        self, stack: Stack, user: Browser, db: psycopg.Connection[Any]
    ) -> None:
        seed_edit(db, 1)
        stack.provider.fail = True
        body = user.post("/v1/explain/1").json()
        assert body["explanation"] == UNAVAILABLE
        assert body["available"] is False
        assert user.get("/v1/queue").status_code == 200

    def test_a_failed_answer_isnt_cached(
        self, stack: Stack, user: Browser, db: psycopg.Connection[Any]
    ) -> None:
        seed_edit(db, 1)
        stack.provider.answer = "Click https://evil.example for details"
        body = user.post("/v1/explain/1").json()
        assert body["reason"] == "rejected_output"
        assert one(db, "SELECT count(*) FROM explanations") == (0,)
        stack.provider.answer = "The edit added a sentence."
        assert user.post("/v1/explain/1").json()["available"] is True

    def test_the_backup_provider_answers_when_the_first_fails(
        self, stack: Stack, user: Browser, db: psycopg.Connection[Any]
    ) -> None:
        from api_test_helpers import FakeProvider

        seed_edit(db, 1)
        backup = FakeProvider("A backup answer.")
        backup.name = "gemini"
        backup.model = "fake-backup"
        stack.app.state.context.explainer.providers.append(backup)
        stack.provider.fail = True
        body = user.post("/v1/explain/1").json()
        assert body["explanation"] == "A backup answer."
        assert body["llm_model"] == "fake-backup"

    def test_a_rejected_answer_doesnt_try_the_backup(
        self, stack: Stack, user: Browser, db: psycopg.Connection[Any]
    ) -> None:
        from api_test_helpers import FakeProvider

        seed_edit(db, 1)
        backup = FakeProvider()
        backup.name = "gemini"
        stack.app.state.context.explainer.providers.append(backup)
        stack.provider.answer = "<script>alert(1)</script>"
        user.post("/v1/explain/1")
        assert backup.calls == []

    def test_concurrent_clicks_share_one_call(
        self, stack: Stack, db: psycopg.Connection[Any]
    ) -> None:
        seed_edit(db, 1)
        browser = stack.browser()
        browser.login()
        gate = threading.Event()
        stack.provider.gate = gate
        results: list[Any] = []

        def click() -> None:
            results.append(browser.post("/v1/explain/1").json())

        threads = [threading.Thread(target=click) for _ in range(4)]
        for thread in threads:
            thread.start()
        deadline = threading.Event()
        while not stack.provider.calls and not deadline.wait(0.01):
            pass
        deadline.wait(0.2)
        gate.set()
        for thread in threads:
            thread.join()
        assert len(stack.provider.calls) == 1
        assert len(results) == 4
        assert all(item["available"] for item in results)

    def test_the_daily_cap_is_durable_and_refuses_past_it(
        self, stack: Stack, user: Browser, db: psycopg.Connection[Any]
    ) -> None:
        seed_edit(db, 1)
        db.execute("INSERT INTO llm_usage (day, provider, calls) VALUES (current_date, 'all', 100)")
        db.commit()
        body = user.post("/v1/explain/1").json()
        assert body["reason"] == "daily_cap"
        assert stack.provider.calls == []

    def test_calls_are_counted_in_postgres(
        self, stack: Stack, user: Browser, db: psycopg.Connection[Any]
    ) -> None:
        seed_edit(db, 1)
        user.post("/v1/explain/1")
        assert one(db, "SELECT calls FROM llm_usage WHERE provider = 'all'") == (1,)
        assert one(db, "SELECT calls FROM llm_usage WHERE provider = 'openai'") == (1,)

    def test_a_cached_answer_still_works_past_the_cap(
        self, stack: Stack, user: Browser, db: psycopg.Connection[Any]
    ) -> None:
        seed_edit(db, 1)
        user.post("/v1/explain/1")
        db.execute("UPDATE llm_usage SET calls = 100 WHERE provider = 'all'")
        db.commit()
        assert user.post("/v1/explain/1").json()["cached"] is True

    def test_a_guest_gets_five_fresh_calls_then_only_cached(
        self, stack: Stack, guest: Browser, db: psycopg.Connection[Any]
    ) -> None:
        for rev_id in range(1, 8):
            seed_edit(db, rev_id)
        for rev_id in range(1, 6):
            assert guest.post(f"/v1/explain/{rev_id}").json()["available"] is True
        refused = guest.post("/v1/explain/6").json()
        assert refused["reason"] == "guest_limit"
        assert len(stack.provider.calls) == 5
        assert guest.post("/v1/explain/1").json()["cached"] is True
        assert guest.get("/v1/auth/me").json()["fresh_explains_left"] == 0

    def test_a_guests_cached_answers_dont_use_up_the_limit(
        self, stack: Stack, guest: Browser, db: psycopg.Connection[Any]
    ) -> None:
        seed_edit(db, 1)
        for _ in range(6):
            assert guest.post("/v1/explain/1").json()["available"] is True
        assert guest.get("/v1/auth/me").json()["fresh_explains_left"] == 4

    def test_explain_has_a_per_ip_limit(
        self, stack: Stack, user: Browser, db: psycopg.Connection[Any]
    ) -> None:
        seed_edit(db, 1)
        statuses = [user.post("/v1/explain/1").status_code for _ in range(11)]
        assert statuses[-1] == 429

    @pytest.mark.parametrize("stack", [{"read_only_mode": True}], indirect=True)
    def test_read_only_mode_allows_cached_but_no_fresh_calls(
        self, stack: Stack, db: psycopg.Connection[Any]
    ) -> None:
        seed_edit(db, 1)
        seed_edit(db, 2)
        db.execute(
            "INSERT INTO explanations (rev_id, prompt_version, llm_model, explanation) "
            "VALUES (1, %s, 'fake-model', 'Cached.')",
            (PROMPT_VERSION,),
        )
        db.commit()
        browser = stack.browser()
        browser.login()
        assert browser.post("/v1/explain/1").json()["explanation"] == "Cached."
        assert browser.post("/v1/explain/2").json()["reason"] == "read_only"
        assert stack.provider.calls == []

    def test_a_changed_prompt_version_or_model_misses_the_cache(
        self, stack: Stack, user: Browser, db: psycopg.Connection[Any]
    ) -> None:
        seed_edit(db, 1)
        db.execute(
            "INSERT INTO explanations (rev_id, prompt_version, llm_model, explanation) "
            "VALUES (1, 'old-version', 'fake-model', 'Old.'), "
            "(1, %s, 'old-model', 'Old too.')",
            (PROMPT_VERSION,),
        )
        db.commit()
        body = user.post("/v1/explain/1").json()
        assert body["cached"] is False
        assert body["explanation"] != "Old."

    def test_the_request_id_reaches_the_provider_interface(
        self, stack: Stack, user: Browser, db: psycopg.Connection[Any]
    ) -> None:
        seed_edit(db, 1)
        seen: list[str | None] = []
        original = stack.provider.complete

        def spy(system: str, prompt: str, request_id: str | None) -> str:
            seen.append(request_id)
            return original(system, prompt, request_id)

        stack.provider.complete = spy  # type: ignore[method-assign]
        user.post("/v1/explain/1", headers={"x-request-id": "explain-request-1"})
        assert seen == ["explain-request-1"]

    def test_llm_outcomes_are_counted(
        self, stack: Stack, user: Browser, db: psycopg.Connection[Any]
    ) -> None:
        seed_edit(db, 1)
        user.post("/v1/explain/1")
        registry = stack.app.state.context.metrics.registry
        assert (
            registry.get_sample_value(
                "api_llm_calls_total", {"provider": "openai", "outcome": "ok"}
            )
            == 1
        )

    def test_nothing_from_the_diff_reaches_the_logs(
        self,
        stack: Stack,
        user: Browser,
        db: psycopg.Connection[Any],
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        seed_edit(db, 1, added_text="UNIQUE-DIFF-MARKER")
        stack.provider.fail = True
        with caplog.at_level("INFO"):
            user.post("/v1/explain/1")
        assert "UNIQUE-DIFF-MARKER" not in caplog.text
