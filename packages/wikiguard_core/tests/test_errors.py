"""What leaves for Sentry: the error and its stack, never the data around it."""

import subprocess
import sys

import sentry_sdk
from pydantic import SecretStr
from wikiguard_core.errors import init_error_tracking, scrub_event, scrub_message


def event() -> dict:
    return {
        "message": "failed for 203.0.113.9",
        "request": {
            "method": "POST",
            "url": "https://api.example/v1/explain/12?token=abc",
            "headers": {"Cookie": "session=secret"},
            "data": {"text": "vandal text"},
            "query_string": "token=abc",
        },
        "user": {"ip_address": "203.0.113.9"},
        "extra": {"diff": "vandal text"},
        "breadcrumbs": {"values": [{"message": "edit by ~2026-12345-6"}]},
        "logentry": {"message": "scoring %s", "params": ["vandal text"]},
        "exception": {
            "values": [
                {
                    "type": "CheckViolation",
                    "value": (
                        'new row violates check\nDETAIL:  Failing row contains (1, "vandal text")'
                    ),
                    "stacktrace": {"frames": [{"function": "score", "vars": {"text": "vandal"}}]},
                }
            ]
        },
    }


def test_request_data_cookies_and_query_strings_are_dropped() -> None:
    scrubbed = scrub_event(event())
    assert scrubbed["request"] == {"method": "POST", "url": "https://api.example/v1/explain/12"}


def test_user_extra_and_breadcrumbs_are_dropped() -> None:
    scrubbed = scrub_event(event())
    assert not {"user", "extra", "breadcrumbs"} & set(scrubbed)


def test_a_quoted_row_and_local_variables_never_leave() -> None:
    scrubbed = scrub_event(event())
    exception = scrubbed["exception"]["values"][0]
    assert exception["value"] == "new row violates check"
    assert "vars" not in exception["stacktrace"]["frames"][0]
    assert "vandal" not in repr(scrubbed)


def test_identifiers_are_redacted_and_messages_cut_short() -> None:
    scrubbed = scrub_event(event())
    assert "203.0.113.9" not in scrubbed["message"]
    assert len(scrub_message("x" * 5000)) == 300
    assert "~2026-12345-6" not in scrub_message("by ~2026-12345-6")


def test_no_dsn_means_off() -> None:
    assert init_error_tracking(None, "api", "dev") is False
    assert init_error_tracking(SecretStr(""), "api", "dev") is False


def test_without_a_dsn_the_sdk_isnt_even_imported() -> None:
    code = (
        "import sys; from wikiguard_core.errors import init_error_tracking; "
        "init_error_tracking(None, 'api', 'dev'); print('sentry_sdk' in sys.modules)"
    )
    # The test's own fixed code, in a fresh interpreter that hasn't imported anything yet.
    result = subprocess.run(  # noqa: S603
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert result.stdout.strip() == "False"


def test_with_a_dsn_events_go_through_the_scrubber() -> None:
    sent: list[dict] = []

    class Capture(sentry_sdk.transport.Transport):
        def capture_envelope(self, envelope) -> None:  # type: ignore[no-untyped-def]
            for item in envelope.items:
                if item.type == "event":
                    sent.append(item.payload.json)

    assert init_error_tracking(SecretStr("https://key@o0.ingest.example/1"), "scorer", "dev")
    sentry_sdk.get_client().transport = Capture()
    # Built at run time: the SDK sends the source lines around a frame, and the test's own source
    # would otherwise hold the words it looks for. Source is public code, not data.
    words = "".join(["van", "dal", " text"])
    try:
        raise ValueError(f"bad row\nDETAIL: Failing row contains ({words})")
    except ValueError:
        sentry_sdk.capture_exception()
    sentry_sdk.flush()
    sentry_sdk.get_client().close()
    assert sent, "nothing captured"
    assert words not in repr(sent)
    assert sent[0]["tags"]["service"] == "scorer"
