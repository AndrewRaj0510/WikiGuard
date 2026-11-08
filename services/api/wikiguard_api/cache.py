"""In-process caching behind one small interface (section 12 of the plan).

- Single flight: when an entry expires, one caller recomputes and the others wait for that answer.
- Stale on error: if recomputing fails, the old value is served and marked stale.
- Jitter: entries don't all expire together.

The LLM call counter and the rate of sign-ins are never kept here, because they must survive a
restart. Answers that depend on who's asking aren't stored here either.
"""

import random
import threading
import time
from collections import OrderedDict
from collections.abc import Callable, Hashable
from concurrent.futures import Future
from dataclasses import dataclass
from typing import Any, Generic, Protocol, TypeVar

T = TypeVar("T")
JITTER_SHARE = 0.1


class CacheMetrics(Protocol):
    def cache_event(self, cache: str, event: str) -> None: ...


class NoMetrics:
    def cache_event(self, cache: str, event: str) -> None:
        return None


@dataclass
class Entry:
    value: Any
    expires_at: float


class SingleFlight(Generic[T]):
    """Concurrent callers with one key share one call, including its failure."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._running: dict[Hashable, Future[T]] = {}

    def do(self, key: Hashable, work: Callable[[], T]) -> tuple[T, bool]:
        """Returns the result, and True if this caller did the work."""
        with self._lock:
            running = self._running.get(key)
            leader = running is None
            if running is None:
                running = Future()
                self._running[key] = running
        if not leader:
            return running.result(), False
        try:
            result = work()
        except BaseException as error:
            running.set_exception(error)
            raise
        else:
            running.set_result(result)
            return result, True
        finally:
            with self._lock:
                del self._running[key]


class TtlCache:
    def __init__(
        self,
        name: str,
        ttl_seconds: float,
        max_entries: int = 256,
        metrics: CacheMetrics | None = None,
        clock: Callable[[], float] = time.monotonic,
        rand: Callable[[], float] = random.random,  # noqa: S311 - timing jitter, not security
    ) -> None:
        self.name = name
        self.ttl = ttl_seconds
        self.max_entries = max_entries
        self.metrics = metrics or NoMetrics()
        self._clock = clock
        self._rand = rand
        self._entries: OrderedDict[Hashable, Entry] = OrderedDict()
        self._lock = threading.Lock()
        self._flight: SingleFlight[Any] = SingleFlight()
        self._generation = 0

    def _expiry(self) -> float:
        return self._clock() + self.ttl * (1 + JITTER_SHARE * self._rand())

    def _store(self, key: Hashable, value: Any, generation: int) -> None:
        with self._lock:
            # A clear() during the load means this value may be out of date, so don't keep it.
            if generation != self._generation:
                return
            self._entries[key] = Entry(value, self._expiry())
            self._entries.move_to_end(key)
            while len(self._entries) > self.max_entries:
                self._entries.popitem(last=False)

    def get(self, key: Hashable, load: Callable[[], Any]) -> tuple[Any, bool]:
        """The value and whether it's stale. Raises only if loading fails and nothing is kept."""
        with self._lock:
            entry = self._entries.get(key)
            generation = self._generation
            if entry is not None:
                self._entries.move_to_end(key)
        if entry is not None and entry.expires_at > self._clock():
            self.metrics.cache_event(self.name, "hit")
            return entry.value, False

        def load_and_store() -> Any:
            value = load()
            self._store(key, value, generation)
            return value

        try:
            value, did_work = self._flight.do(key, load_and_store)
        except Exception:
            if entry is None:
                self.metrics.cache_event(self.name, "error")
                raise
            self.metrics.cache_event(self.name, "stale")
            return entry.value, True
        self.metrics.cache_event(self.name, "miss" if did_work else "shared")
        return value, False

    def clear(self) -> None:
        with self._lock:
            self._generation += 1
            self._entries.clear()
