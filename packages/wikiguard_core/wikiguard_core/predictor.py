"""The serving chain: try models in order, and keep answering when one of them breaks.

This file imports no ONNX Runtime. It sees models only through the Scorer protocol, so its failure
rules can be tested with fakes, and a ServedModel fits without any adapter.

Rules that aren't obvious, all recorded in docs/decision.md (DEC-054):

- Each model has its own worker slots. With a shared pool, a hung champion would take every slot
  and block the MLP behind it as well, so one bad model would stop the whole chain.
- ONNX inference can't be cancelled. A timed-out call keeps running and keeps its slot until it
  finishes, which is what bounds the number of stuck threads. When the slots are full, the next
  call moves on immediately and doesn't queue.
- Workers are daemon threads, not a ThreadPoolExecutor. An executor queues work, and Python joins
  its threads at exit, so one hung call would stop the process from shutting down.
- An exception counts against a model only if a later model answered the same rows. A malformed row
  raises in every model, and without this rule one bad event would open every breaker. Timeouts and
  full slots always count, because a poison row can't cause those and overload must open breakers.
- The last model is tried even when its breaker is open. Nothing stands behind it, so skipping it
  could only turn a slow answer into no answer.
"""

import logging
import threading
import time
from collections.abc import Callable, Iterable, Sequence
from concurrent.futures import Future, wait
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from wikiguard_core.inputs import Row

logger = logging.getLogger(__name__)

BREAKER_THRESHOLD = 3
BREAKER_COOLDOWN_SECONDS = 60.0
# One for a running call and one spare, so a single timed-out call doesn't block the next.
DEFAULT_POOL_SIZE = 2
MS_PER_SECOND = 1000.0


class Scorer(Protocol):
    threshold: float | None

    def probabilities(self, rows: Sequence[Row]) -> NDArray[np.float64]: ...


class Outcome(StrEnum):
    ANSWERED = "answered"
    FAILED = "failed"
    TIMEOUT = "timeout"
    BUSY = "busy"
    SKIPPED = "skipped"


class BreakerState(StrEnum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


@dataclass(frozen=True)
class ChainEntry:
    alias: str
    model_name: str
    model_version: str
    model: Scorer


@dataclass(frozen=True)
class Attempt:
    alias: str
    model_name: str
    outcome: Outcome
    latency_ms: float = 0.0
    # The exception's type only. Its message could quote a diff, and this ends up in logs.
    detail: str | None = None


@dataclass(frozen=True)
class Prediction:
    """Scores in the order of the rows. One model answers for the whole batch."""

    scores: NDArray[np.float64]
    alias: str
    model_name: str
    model_version: str
    threshold: float | None
    used_fallback: bool
    forced: bool
    latency_ms: float
    attempts: tuple[Attempt, ...]


@dataclass(frozen=True)
class ModelStatus:
    alias: str
    model_name: str
    model_version: str
    breaker: BreakerState
    consecutive_failures: int


class AllModelsFailed(RuntimeError):
    """No model answered. The caller decides what that means: retry, or send to the DLQ."""

    def __init__(self, attempts: Iterable[Attempt]) -> None:
        self.attempts = tuple(attempts)
        summary = ", ".join(f"{item.alias}={item.outcome}" for item in self.attempts)
        super().__init__(f"no model answered: {summary}")


class ModelOutputError(ValueError):
    """A model returned something that can't be a score, such as NaN."""


class _SlotsFull(Exception):
    pass


class _TimedOut(Exception):
    pass


class CircuitBreaker:
    """Stops calling a broken model for a while, then lets one trial call through."""

    def __init__(
        self,
        threshold: int = BREAKER_THRESHOLD,
        cooldown_seconds: float = BREAKER_COOLDOWN_SECONDS,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._threshold = threshold
        self._cooldown = cooldown_seconds
        self._clock = clock
        self._lock = threading.Lock()
        self._failures = 0
        self._opened_at: float | None = None
        self._trial_running = False

    def allow(self) -> bool:
        """Whether to call the model. A True after the cooldown holds the single trial."""
        with self._lock:
            if self._opened_at is None:
                return True
            if self._clock() - self._opened_at < self._cooldown or self._trial_running:
                return False
            self._trial_running = True
            return True

    def success(self) -> None:
        with self._lock:
            self._failures = 0
            self._opened_at = None
            self._trial_running = False

    def failure(self) -> bool:
        """Record a failure, and return True if this one opened the breaker."""
        with self._lock:
            self._failures += 1
            reopening = self._trial_running
            self._trial_running = False
            was_closed = self._opened_at is None
            if reopening or (was_closed and self._failures >= self._threshold):
                self._opened_at = self._clock()
                return True
            return False

    def release(self) -> None:
        """End a trial with no verdict, because the rows may have been at fault, not the model."""
        with self._lock:
            self._trial_running = False

    @property
    def state(self) -> BreakerState:
        with self._lock:
            if self._opened_at is None:
                return BreakerState.CLOSED
            if self._clock() - self._opened_at < self._cooldown:
                return BreakerState.OPEN
            return BreakerState.HALF_OPEN

    @property
    def consecutive_failures(self) -> int:
        with self._lock:
            return self._failures


class _Link:
    """One model in the chain, with its own breaker and worker slots."""

    def __init__(self, entry: ChainEntry, pool_size: int, breaker: CircuitBreaker) -> None:
        self.entry = entry
        self.breaker = breaker
        self._slots = threading.BoundedSemaphore(pool_size)

    def _work(self, rows: Sequence[Row], future: "Future[NDArray[np.float64]]") -> None:
        try:
            future.set_result(self.entry.model.probabilities(rows))
        except Exception as error:  # noqa: BLE001 - handed to the waiting caller, not swallowed
            future.set_exception(error)
        finally:
            self._slots.release()

    def run(self, rows: Sequence[Row], timeout_seconds: float) -> NDArray[np.float64]:
        if not self._slots.acquire(blocking=False):
            raise _SlotsFull
        future: Future[NDArray[np.float64]] = Future()
        thread = threading.Thread(
            target=self._work,
            args=(rows, future),
            name=f"predict-{self.entry.alias}",
            daemon=True,
        )
        try:
            thread.start()
        except RuntimeError:
            self._slots.release()
            raise
        if not wait([future], timeout_seconds).done:
            raise _TimedOut
        return _checked(future.result(), len(rows))


def _checked(scores: NDArray[np.float64], expected: int) -> NDArray[np.float64]:
    """A NaN would otherwise be stored as a score and sort unpredictably in the review queue."""
    result = np.asarray(scores, dtype=np.float64).reshape(-1)
    if result.size != expected:
        raise ModelOutputError(f"expected {expected} scores, got {result.size}")
    if not np.all(np.isfinite(result)) or np.any(result < 0.0) or np.any(result > 1.0):
        raise ModelOutputError("score outside 0 to 1")
    return result


class Predictor:
    def __init__(
        self,
        chain: Sequence[ChainEntry],
        timeout_ms: int,
        force_model: str | None = None,
        pool_size: int = DEFAULT_POOL_SIZE,
        breaker_threshold: int = BREAKER_THRESHOLD,
        breaker_cooldown_seconds: float = BREAKER_COOLDOWN_SECONDS,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if not chain:
            raise ValueError("the chain needs at least one model")
        if timeout_ms <= 0:
            raise ValueError("timeout_ms must be positive")
        self._timeout_ms = timeout_ms
        self._forced = force_model is not None
        if force_model is not None:
            chain = self._only_forced(chain, force_model)
        self._links = [
            _Link(
                entry,
                pool_size,
                CircuitBreaker(breaker_threshold, breaker_cooldown_seconds, clock),
            )
            for entry in chain
        ]

    @staticmethod
    def _only_forced(chain: Sequence[ChainEntry], force_model: str) -> list[ChainEntry]:
        """The kill switch runs one model with nothing behind it: traffic must never quietly reach
        a model the owner just turned off. Failing at startup beats ignoring the switch."""
        chosen = [entry for entry in chain if entry.model_name == force_model]
        if not chosen:
            loaded = sorted(entry.model_name for entry in chain)
            raise ValueError(f"FORCE_MODEL={force_model} isn't loaded, this chain has {loaded}")
        return chosen[:1]

    def budget_seconds(self, count: int) -> float:
        """PREDICT_TIMEOUT_MS is per edit, so a micro-batch gets the sum of its edits' budgets."""
        return self._timeout_ms * count / MS_PER_SECOND

    def status(self) -> list[ModelStatus]:
        return [
            ModelStatus(
                link.entry.alias,
                link.entry.model_name,
                link.entry.model_version,
                link.breaker.state,
                link.breaker.consecutive_failures,
            )
            for link in self._links
        ]

    def _answer(
        self,
        link: _Link,
        position: int,
        scores: NDArray[np.float64],
        started: float,
        attempts: list[Attempt],
    ) -> Prediction:
        return Prediction(
            scores=scores,
            alias=link.entry.alias,
            model_name=link.entry.model_name,
            model_version=link.entry.model_version,
            threshold=link.entry.model.threshold,
            used_fallback=position > 0,
            forced=self._forced,
            latency_ms=(time.perf_counter() - started) * MS_PER_SECOND,
            attempts=tuple(attempts),
        )

    @staticmethod
    def _record_failure(link: _Link) -> None:
        if link.breaker.failure():
            logger.warning("circuit opened for %s (%s)", link.entry.alias, link.entry.model_name)

    def score(self, rows: Sequence[Row]) -> Prediction:
        """Scores from the first model that answers. Raises AllModelsFailed if none does."""
        started = time.perf_counter()
        attempts: list[Attempt] = []
        if not rows:
            return self._answer(self._links[0], 0, np.zeros(0), started, attempts)
        raised: list[_Link] = []
        budget = self.budget_seconds(len(rows))
        for position, link in enumerate(self._links):
            entry = link.entry
            is_last = position == len(self._links) - 1
            if not link.breaker.allow() and not is_last:
                attempts.append(Attempt(entry.alias, entry.model_name, Outcome.SKIPPED))
                continue
            attempt_started = time.perf_counter()
            outcome = Outcome.FAILED
            detail: str | None = None
            try:
                scores = link.run(rows, budget)
            except _SlotsFull:
                outcome = Outcome.BUSY
                self._record_failure(link)
            except _TimedOut:
                outcome = Outcome.TIMEOUT
                self._record_failure(link)
            except Exception as error:  # noqa: BLE001 - any model failure moves down the chain
                detail = type(error).__name__
                raised.append(link)
            else:
                elapsed = (time.perf_counter() - attempt_started) * MS_PER_SECOND
                attempts.append(Attempt(entry.alias, entry.model_name, Outcome.ANSWERED, elapsed))
                link.breaker.success()
                for earlier in raised:
                    self._record_failure(earlier)
                return self._answer(link, position, scores, started, attempts)
            elapsed = (time.perf_counter() - attempt_started) * MS_PER_SECOND
            attempts.append(Attempt(entry.alias, entry.model_name, outcome, elapsed, detail))
        for link in raised:
            link.breaker.release()
        failure = AllModelsFailed(attempts)
        logger.error("%s", failure)
        raise failure
