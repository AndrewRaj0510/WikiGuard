"""The chain's failure rules, with fake models and a fake clock. No ONNX Runtime needed."""

import logging
import subprocess
import sys
import textwrap
import threading
import time
from collections.abc import Callable, Iterator, Sequence
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray
from wikiguard_core.config import ServingSettings
from wikiguard_core.predictor import (
    AllModelsFailed,
    BreakerState,
    ChainEntry,
    CircuitBreaker,
    Outcome,
    Predictor,
)

ROW: dict[str, Any] = {"comment": "ok"}
POISON: dict[str, Any] = {"poison": True}
# Long enough that an unexpected timeout means a real bug, even on a busy CI machine.
PATIENT_MS = 5000
SHORT_MS = 50


class FakeClock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class FakeModel:
    def __init__(
        self,
        score: float = 0.5,
        error: Exception | None = None,
        gate: threading.Event | None = None,
        delay: float = 0.0,
        output: Any = None,
    ) -> None:
        self.score = score
        self.error = error
        self.gate = gate
        self.delay = delay
        self.output = output
        self.threshold: float | None = 0.9
        self.calls = 0
        self._lock = threading.Lock()

    def probabilities(self, rows: Sequence[dict[str, Any]]) -> NDArray[np.float64]:
        with self._lock:
            self.calls += 1
        if self.gate is not None:
            self.gate.wait()
        if self.delay:
            time.sleep(self.delay)
        if any(row.get("poison") for row in rows):
            raise KeyError("hour_of_day")
        if self.error is not None:
            raise self.error
        if self.output is not None:
            return np.asarray(self.output, dtype=np.float64)
        return np.full(len(rows), self.score, dtype=np.float64)


def chain_of(*models: FakeModel) -> list[ChainEntry]:
    names = ["roberta", "distilbert", "bilstm", "bert", "mlp"]
    aliases = ["champion", "fallback-1", "fallback-2", "fallback-3", "fallback-4"]
    # The last model passed is always the MLP, as in the real chain.
    names = [*names[: len(models) - 1], "mlp"]
    return [
        ChainEntry(aliases[index], names[index], str(index + 1), model)
        for index, model in enumerate(models)
    ]


@pytest.fixture
def gate() -> Iterator[threading.Event]:
    """Holds a fake model in a hung call, and lets it finish when the test ends."""
    event = threading.Event()
    yield event
    event.set()


def state_of(predictor: Predictor, alias: str) -> BreakerState:
    return next(item.breaker for item in predictor.status() if item.alias == alias)


# --- The chain ---


def test_champion_answers_with_no_fallback() -> None:
    champion, mlp = FakeModel(score=0.8), FakeModel(score=0.1)
    result = Predictor(chain_of(champion, mlp), PATIENT_MS).score([ROW, ROW])
    assert result.scores.tolist() == [0.8, 0.8]
    assert (result.alias, result.model_name, result.model_version) == ("champion", "roberta", "1")
    assert result.used_fallback is False
    assert result.forced is False
    assert result.threshold == 0.9
    assert result.latency_ms >= 0
    assert [item.outcome for item in result.attempts] == [Outcome.ANSWERED]
    assert mlp.calls == 0


def test_failing_champion_falls_back_and_says_so() -> None:
    champion, fallback, mlp = (
        FakeModel(error=RuntimeError("broken")),
        FakeModel(0.3),
        FakeModel(0.1),
    )
    result = Predictor(chain_of(champion, fallback, mlp), PATIENT_MS).score([ROW])
    assert result.scores.tolist() == [0.3]
    assert result.model_name == "distilbert"
    assert result.used_fallback is True
    assert [item.outcome for item in result.attempts] == [Outcome.FAILED, Outcome.ANSWERED]
    assert mlp.calls == 0


def test_chain_is_tried_in_order_down_to_the_mlp() -> None:
    models = [FakeModel(error=RuntimeError("broken")) for _ in range(4)]
    result = Predictor(chain_of(*models, FakeModel(0.1)), PATIENT_MS).score([ROW])
    assert result.model_name == "mlp"
    assert [item.alias for item in result.attempts] == [
        "champion",
        "fallback-1",
        "fallback-2",
        "fallback-3",
        "fallback-4",
    ]


def test_every_model_failing_raises_with_the_attempts() -> None:
    chain = chain_of(FakeModel(error=RuntimeError("a")), FakeModel(error=RuntimeError("b")))
    with pytest.raises(AllModelsFailed, match="champion=failed") as caught:
        Predictor(chain, PATIENT_MS).score([ROW])
    assert [item.outcome for item in caught.value.attempts] == [Outcome.FAILED, Outcome.FAILED]


def test_empty_batch_calls_no_model() -> None:
    champion = FakeModel()
    result = Predictor(chain_of(champion, FakeModel()), PATIENT_MS).score([])
    assert result.scores.size == 0
    assert champion.calls == 0


@pytest.mark.parametrize(
    "output",
    [[float("nan")], [float("inf")], [1.5], [-0.1], [0.5, 0.5]],
    ids=["nan", "infinity", "above-one", "below-zero", "wrong-length"],
)
def test_impossible_output_is_a_failure_not_a_score(output: list[float]) -> None:
    champion, mlp = FakeModel(output=output), FakeModel(0.1)
    result = Predictor(chain_of(champion, mlp), PATIENT_MS).score([ROW])
    assert result.model_name == "mlp"
    assert result.attempts[0].detail == "ModelOutputError"


def test_an_empty_chain_and_a_bad_timeout_are_refused() -> None:
    with pytest.raises(ValueError, match="at least one"):
        Predictor([], PATIENT_MS)
    with pytest.raises(ValueError, match="positive"):
        Predictor(chain_of(FakeModel()), 0)


def test_failure_details_never_contain_the_rows_text(caplog: pytest.LogCaptureFixture) -> None:
    secret = "vandal text from a diff"
    chain = chain_of(FakeModel(error=ValueError(secret)), FakeModel(error=ValueError(secret)))
    with caplog.at_level(logging.DEBUG), pytest.raises(AllModelsFailed) as caught:
        Predictor(chain, PATIENT_MS).score([ROW])
    assert secret not in str(caught.value)
    assert secret not in caplog.text
    assert caught.value.attempts[0].detail == "ValueError"


# --- Timeouts and worker slots ---


def test_hung_champion_times_out_and_falls_back(gate: threading.Event) -> None:
    champion, mlp = FakeModel(gate=gate), FakeModel(0.1)
    started = time.perf_counter()
    result = Predictor(chain_of(champion, mlp), SHORT_MS).score([ROW])
    assert time.perf_counter() - started < 2
    assert result.model_name == "mlp"
    assert result.attempts[0].outcome == Outcome.TIMEOUT


def test_budget_scales_with_the_batch() -> None:
    predictor = Predictor(chain_of(FakeModel()), 300)
    assert predictor.budget_seconds(1) == pytest.approx(0.3)
    assert predictor.budget_seconds(32) == pytest.approx(9.6)


def test_a_batch_gets_a_bigger_budget_than_one_edit() -> None:
    slow, mlp = FakeModel(delay=0.2), FakeModel(0.1)
    predictor = Predictor(chain_of(slow, mlp), 100, pool_size=8)
    assert predictor.score([ROW]).model_name == "mlp"
    assert predictor.score([ROW] * 50).model_name == "roberta"


def test_full_slots_move_on_at_once_and_never_queue(gate: threading.Event) -> None:
    champion, mlp = FakeModel(gate=gate), FakeModel(0.1)
    predictor = Predictor(chain_of(champion, mlp), PATIENT_MS, pool_size=1)
    first: list[str] = []
    waiting = threading.Thread(target=lambda: first.append(predictor.score([ROW]).model_name))
    waiting.start()
    deadline = time.monotonic() + 5
    while champion.calls == 0 and time.monotonic() < deadline:
        time.sleep(0.001)
    # The first call is inside the model and holds the only slot, with five seconds to run.
    started = time.perf_counter()
    result = predictor.score([ROW])
    assert time.perf_counter() - started < 2
    assert result.attempts[0].outcome == Outcome.BUSY
    assert result.model_name == "mlp"
    gate.set()
    waiting.join()
    assert first == ["roberta"]


def test_a_hung_champion_leaves_the_mlp_able_to_run(gate: threading.Event) -> None:
    champion, mlp = FakeModel(gate=gate), FakeModel(0.1)
    predictor = Predictor(chain_of(champion, mlp), SHORT_MS, pool_size=1, breaker_threshold=100)
    for _ in range(5):
        assert predictor.score([ROW]).model_name == "mlp"
    assert champion.calls == 1  # the one stuck call holds the slot, and no more threads start
    assert mlp.calls == 5


def test_a_slot_comes_back_when_a_stuck_call_finishes(gate: threading.Event) -> None:
    champion, mlp = FakeModel(gate=gate), FakeModel(0.1)
    predictor = Predictor(chain_of(champion, mlp), SHORT_MS, pool_size=1, breaker_threshold=100)
    predictor.score([ROW])
    gate.set()
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        if predictor.score([ROW]).model_name == "roberta":
            return
    pytest.fail("the slot never came back")


def test_a_hung_model_does_not_stop_the_process_exiting() -> None:
    script = textwrap.dedent(
        """
        import threading
        import numpy as np
        from wikiguard_core.predictor import ChainEntry, Predictor

        class Hung:
            threshold = None
            def probabilities(self, rows):
                threading.Event().wait()

        class Answers:
            threshold = None
            def probabilities(self, rows):
                return np.zeros(len(rows))

        chain = [
            ChainEntry("champion", "a", "1", Hung()),
            ChainEntry("fallback-1", "b", "2", Answers()),
        ]
        print(Predictor(chain, 50).score([{}]).model_name)
        """
    )
    done = subprocess.run(  # noqa: S603 - this interpreter, running the script above
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=30, check=True
    )
    assert done.stdout.strip() == "b"


# --- The circuit breaker ---


def broken_chain(clock: FakeClock) -> tuple[Predictor, FakeModel, FakeModel]:
    champion, mlp = FakeModel(error=RuntimeError("broken")), FakeModel(0.1)
    return Predictor(chain_of(champion, mlp), PATIENT_MS, clock=clock), champion, mlp


def test_breaker_skips_a_model_after_three_failures() -> None:
    predictor, champion, _ = broken_chain(FakeClock())
    for _ in range(3):
        assert predictor.score([ROW]).attempts[0].outcome == Outcome.FAILED
    assert state_of(predictor, "champion") == BreakerState.OPEN
    result = predictor.score([ROW])
    assert result.attempts[0].outcome == Outcome.SKIPPED
    assert result.used_fallback is True
    assert champion.calls == 3


def test_breaker_stays_open_for_the_whole_cooldown() -> None:
    clock = FakeClock()
    predictor, champion, _ = broken_chain(clock)
    for _ in range(3):
        predictor.score([ROW])
    clock.advance(59.9)
    predictor.score([ROW])
    assert champion.calls == 3


def test_failed_trial_reopens_for_a_full_cooldown() -> None:
    clock = FakeClock()
    predictor, champion, _ = broken_chain(clock)
    for _ in range(3):
        predictor.score([ROW])
    clock.advance(60)
    assert state_of(predictor, "champion") == BreakerState.HALF_OPEN
    predictor.score([ROW])
    assert champion.calls == 4
    assert state_of(predictor, "champion") == BreakerState.OPEN
    clock.advance(59.9)
    predictor.score([ROW])
    assert champion.calls == 4


def test_successful_trial_closes_the_breaker() -> None:
    clock = FakeClock()
    predictor, champion, _ = broken_chain(clock)
    for _ in range(3):
        predictor.score([ROW])
    champion.error = None
    clock.advance(60)
    assert predictor.score([ROW]).model_name == "roberta"
    assert state_of(predictor, "champion") == BreakerState.CLOSED
    assert predictor.status()[0].consecutive_failures == 0


def test_a_success_resets_the_count_so_failures_must_be_consecutive() -> None:
    predictor, champion, _ = broken_chain(FakeClock())
    for _ in range(2):
        predictor.score([ROW])
    champion.error = None
    predictor.score([ROW])
    champion.error = RuntimeError("broken")
    for _ in range(2):
        predictor.score([ROW])
    assert state_of(predictor, "champion") == BreakerState.CLOSED


def test_only_one_trial_runs_after_the_cooldown() -> None:
    clock = FakeClock()
    breaker = CircuitBreaker(clock=clock)
    for _ in range(3):
        breaker.failure()
    clock.advance(60)
    assert breaker.allow() is True
    assert breaker.allow() is False


def test_timeouts_open_the_breaker(gate: threading.Event) -> None:
    champion, mlp = FakeModel(gate=gate), FakeModel(0.1)
    predictor = Predictor(chain_of(champion, mlp), SHORT_MS, pool_size=8)
    for _ in range(3):
        predictor.score([ROW])
    assert state_of(predictor, "champion") == BreakerState.OPEN
    assert predictor.score([ROW]).attempts[0].outcome == Outcome.SKIPPED


def test_timeouts_count_even_when_every_model_times_out(gate: threading.Event) -> None:
    """Overload hits the whole chain, and the breakers must still open to shed the waiting."""
    chain = chain_of(FakeModel(gate=gate), FakeModel(gate=gate))
    predictor = Predictor(chain, SHORT_MS, pool_size=8)
    for _ in range(3):
        with pytest.raises(AllModelsFailed):
            predictor.score([ROW])
    assert state_of(predictor, "champion") == BreakerState.OPEN


def test_the_last_model_is_tried_even_with_its_breaker_open(gate: threading.Event) -> None:
    champion, mlp = FakeModel(error=RuntimeError("broken")), FakeModel(gate=gate)
    predictor = Predictor(chain_of(champion, mlp), SHORT_MS, pool_size=8)
    for _ in range(3):
        with pytest.raises(AllModelsFailed):
            predictor.score([ROW])
    assert state_of(predictor, "fallback-1") == BreakerState.OPEN
    # Well inside the cooldown, and it's still called, and its success closes the breaker.
    gate.set()
    assert predictor.score([ROW]).model_name == "mlp"
    assert state_of(predictor, "fallback-1") == BreakerState.CLOSED


# --- A malformed row isn't a broken model ---


def test_a_poison_row_opens_no_breakers() -> None:
    champion, mlp = FakeModel(0.8), FakeModel(0.1)
    predictor = Predictor(chain_of(champion, mlp), PATIENT_MS)
    for _ in range(10):
        with pytest.raises(AllModelsFailed):
            predictor.score([POISON])
    assert [item.breaker for item in predictor.status()] == [BreakerState.CLOSED] * 2
    assert predictor.score([ROW]).model_name == "roberta"


def test_poison_rows_during_a_trial_leave_the_breaker_able_to_try_again() -> None:
    clock = FakeClock()
    predictor, champion, _ = broken_chain(clock)
    for _ in range(3):
        predictor.score([ROW])
    clock.advance(60)
    champion.error = None
    with pytest.raises(AllModelsFailed):
        predictor.score([POISON])
    assert predictor.score([ROW]).model_name == "roberta"


def test_a_failure_only_the_champion_has_still_counts() -> None:
    predictor, _, _ = broken_chain(FakeClock())
    predictor.score([ROW])
    assert predictor.status()[0].consecutive_failures == 1


# --- FORCE_MODEL ---


def test_force_model_uses_only_that_model() -> None:
    champion, mlp = FakeModel(0.8), FakeModel(0.1)
    result = Predictor(chain_of(champion, mlp), PATIENT_MS, force_model="mlp").score([ROW])
    assert result.model_name == "mlp"
    assert result.forced is True
    assert result.used_fallback is False
    assert champion.calls == 0


def test_a_failing_forced_model_raises_and_never_falls_back() -> None:
    champion, mlp = FakeModel(0.8), FakeModel(error=RuntimeError("broken"))
    predictor = Predictor(chain_of(champion, mlp), PATIENT_MS, force_model="mlp")
    with pytest.raises(AllModelsFailed):
        predictor.score([ROW])
    assert champion.calls == 0


def test_force_model_not_loaded_fails_at_startup() -> None:
    with pytest.raises(ValueError, match="FORCE_MODEL=bilstm isn't loaded"):
        Predictor(chain_of(FakeModel(), FakeModel()), PATIENT_MS, force_model="bilstm")


def test_a_blank_force_model_setting_means_no_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FORCE_MODEL", "")
    settings = ServingSettings(_env_file=None)  # type: ignore[call-arg]
    assert settings.force_model is None
    result = Predictor(
        chain_of(FakeModel(0.8), FakeModel(0.1)),
        settings.predict_timeout_ms,
        force_model=settings.force_model,
    ).score([ROW])
    assert result.model_name == "roberta"
    assert result.forced is False


# --- Many callers at once ---


def run_at_once(calls: int, workers: int, call: Callable[[], None]) -> list[Exception]:
    errors: list[Exception] = []

    def work() -> None:
        for _ in range(calls):
            try:
                call()
            except Exception as error:  # noqa: BLE001 - the test reports whatever went wrong
                errors.append(error)

    threads = [threading.Thread(target=work) for _ in range(workers)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    return errors


def test_concurrent_calls_with_a_failing_champion_all_get_an_answer() -> None:
    champion, mlp = FakeModel(error=RuntimeError("broken")), FakeModel(0.1)
    predictor = Predictor(chain_of(champion, mlp), PATIENT_MS, pool_size=8)
    answers: list[str] = []

    def call() -> None:
        answers.append(predictor.score([ROW]).model_name)

    assert run_at_once(25, 8, call) == []
    assert answers == ["mlp"] * 200
    assert state_of(predictor, "champion") == BreakerState.OPEN
    # Calls already in flight when it opened may add a few, but the breaker stops the rest.
    assert champion.calls < 20
