"""An API on real Postgres, logged in as the real wikiguard_api role, with fake models and LLM.

Sessions, the seat, and the counters are rules written in SQL, so a fake database would test
nothing.
"""

import json
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import psycopg
from argon2 import PasswordHasher
from fastapi import FastAPI
from fastapi.testclient import TestClient
from wikiguard_api.app import SESSION_COOKIE, create_app
from wikiguard_api.explain import ProviderError
from wikiguard_api.foundations import Settings
from wikiguard_core.artifacts import write_checksums
from wikiguard_core.chain import ModelCache, ModelRef, ServingChain
from wikiguard_core.features import FEATURE_VERSION

PASSWORD = "correct horse battery staple"
# Cheap settings for tests. Verifying reads them from the hash.
PASSWORD_HASH = PasswordHasher(time_cost=1, memory_cost=8, parallelism=1).hash(PASSWORD)
SECRET_KEY = "x-test-only-x-test-only-x-test-only-x-test-only"
METRICS_TOKEN = "metrics-test-token"


class FakeModel:
    threshold: float | None = 0.9

    def __init__(self, score: float = 0.8) -> None:
        self.score = score
        self.calls = 0
        self.fail = False

    def probabilities(self, rows: Any) -> Any:
        self.calls += 1
        if self.fail:
            raise RuntimeError("broken")
        return np.full(len(rows), self.score)


class FakeSource:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.refs: list[ModelRef] = []
        for index, name in enumerate(("roberta", "mlp")):
            folder = root / "store" / name
            folder.mkdir(parents=True)
            (folder / "config.json").write_text(json.dumps({"feature_version": FEATURE_VERSION}))
            (folder / "model.onnx").write_bytes(name.encode())
            alias = "champion" if index == 0 else "fallback-1"
            self.refs.append(ModelRef(alias, name, str(index + 1), write_checksums(folder)))

    def resolve(self) -> list[ModelRef]:
        return list(self.refs)

    def fetch(self, ref: ModelRef, target: Path) -> None:
        for path in (self.root / "store" / ref.model_name).iterdir():
            (target / path.name).write_bytes(path.read_bytes())


class FakeProvider:
    name = "openai"
    model = "fake-model"

    def __init__(self, answer: str = "The edit added a sentence about the Moon.") -> None:
        self.answer = answer
        self.calls: list[str] = []
        self.fail = False
        self.gate: Any = None

    def complete(self, system: str, user: str, request_id: str | None) -> str:
        self.calls.append(user)
        if self.gate is not None:
            self.gate.wait(10)
        if self.fail:
            raise ProviderError("status 500")
        return self.answer


class Browser:
    """One browser profile: its own cookies, and a tab ID per page load."""

    def __init__(self, app: FastAPI, tab: str = "tab-aaaaaaaa", **options: Any) -> None:
        self.client = TestClient(app, **options)
        self.tab = tab
        self.csrf = ""

    def headers(self, extra: dict[str, str] | None = None) -> dict[str, str]:
        return {"x-tab-id": self.tab, "x-csrf-token": self.csrf} | (extra or {})

    def guest(self) -> Any:
        response = self.client.post("/v1/auth/guest", json={"tab_id": self.tab})
        if response.status_code == 200:
            self.csrf = response.json()["csrf_token"]
        return response

    def login(self, password: str = PASSWORD, take_over: bool = False) -> Any:
        response = self.client.post(
            "/v1/auth/login",
            json={"password": password, "tab_id": self.tab, "take_over": take_over},
        )
        if response.status_code == 200:
            self.csrf = response.json()["csrf_token"]
        return response

    def get(self, path: str, **options: Any) -> Any:
        headers = self.headers(options.pop("headers", None))
        return self.client.get(path, headers=headers, **options)

    def post(self, path: str, **options: Any) -> Any:
        headers = self.headers(options.pop("headers", None))
        return self.client.post(path, headers=headers, **options)

    def put(self, path: str, **options: Any) -> Any:
        headers = self.headers(options.pop("headers", None))
        return self.client.put(path, headers=headers, **options)

    def heartbeat(self, active: bool = False) -> Any:
        return self.post("/v1/gate/heartbeat", json={"active": active})

    def cookie(self) -> str | None:
        return self.client.cookies.get(SESSION_COOKIE)


def make_settings(url: str, **overrides: Any) -> Settings:
    values: dict[str, Any] = {
        "environment": "dev",
        "database_url": url,
        "session_secret_key": SECRET_KEY,
        "user_password_hash": PASSWORD_HASH,
        "metrics_bearer_token": METRICS_TOKEN,
        "model_poll_seconds": 0,
        "model_source": "local",
        "predict_timeout_ms": 5000,
    }
    return Settings(_env_file=None, **(values | overrides))  # type: ignore[call-arg]


class Stack:
    """The app, its fakes, and a way to wait until models are in."""

    def __init__(
        self, url: str, tmp_path: Path, make: Callable[..., FastAPI] = create_app, **overrides: Any
    ) -> None:
        self.settings = make_settings(url, **overrides)
        self.models = {"roberta": FakeModel(0.8), "mlp": FakeModel(0.2)}
        self.provider = FakeProvider()
        self.chain = ServingChain(
            FakeSource(tmp_path),
            ModelCache(tmp_path / "cache"),
            self.settings.predict_timeout_ms,
            self.settings.force_model,
            loader=lambda folder, _: self.models[folder.name.split("-")[0]],
        )
        self.statements: list[str] = []
        self.app = make(
            self.settings, self.chain, [self.provider], 100, on_statement=self.statements.append
        )

    def wait_for_models(self) -> None:
        deadline = datetime.now(UTC).timestamp() + 10
        while not self.chain.ready and datetime.now(UTC).timestamp() < deadline:
            time.sleep(0.002)
        assert self.chain.ready
        for model in self.models.values():
            model.calls = 0

    def browser(self, tab: str = "tab-aaaaaaaa", **options: Any) -> Browser:
        return Browser(self.app, tab, **options)


def seed_edit(
    db: psycopg.Connection[Any],
    rev_id: int,
    score: float | None = 0.5,
    is_anonymous: bool = False,
    edited_at: datetime | None = None,
    hidden: bool = False,
    version: str = "1",
    added_text: str | None = "added words",
) -> None:
    moment = edited_at or datetime.now(UTC)
    db.execute(
        """
        INSERT INTO edits (rev_id, parent_rev_id, schema_version, title, edited_at, is_minor,
            is_anonymous, user_hash, comment, old_length, new_length, added_text, removed_text,
            user_edit_count, content_hidden, latest_score, latest_model_version, latest_scored_at)
        VALUES (%s, %s, 1, 'Moon', %s, false, %s, 'SECRET-HASH', 'a comment', 100, 120, %s,
            'removed words', 3, %s, %s, %s, %s)
        """,
        (
            rev_id,
            rev_id - 1,
            moment,
            is_anonymous,
            None if hidden else added_text,
            hidden,
            score,
            version if score is not None else None,
            moment if score is not None else None,
        ),
    )
    if score is not None:
        db.execute(
            "INSERT INTO scores (rev_id, model_name, model_version, score, alias, threshold) "
            "VALUES (%s, 'roberta', %s, %s, 'champion', 0.9)",
            (rev_id, version, score),
        )
    db.commit()


def one(db: psycopg.Connection[Any], query: str, *params: Any) -> Any:
    row = db.execute(query, params).fetchone()
    db.commit()
    return row
