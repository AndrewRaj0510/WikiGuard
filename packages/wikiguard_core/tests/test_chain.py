"""Aliases, the verified cache, and safe reloads, with real files and no ONNX Runtime."""

import json
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import pytest
from wikiguard_core.artifacts import ArtifactError, write_checksums
from wikiguard_core.chain import (
    ChainError,
    ChainMode,
    LocalRuns,
    MlflowRegistry,
    ModelCache,
    ModelRef,
    ServingChain,
    source_from_settings,
)
from wikiguard_core.config import ServingSettings
from wikiguard_core.features import FEATURE_VERSION
from wikiguard_core.predictor import Scorer

FAST_TIMEOUT_MS = 5000


def make_artifact(folder: Path, marker: str = "x") -> str:
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "config.json").write_text(
        json.dumps({"feature_version": FEATURE_VERSION, "kind": "mlp", "marker": marker}), "utf-8"
    )
    (folder / "model.onnx").write_bytes(marker.encode() * 100)
    return write_checksums(folder)


class FakeModel:
    threshold: float | None = 0.5

    def __init__(self, marker: str, broken: bool = False) -> None:
        self.marker = marker
        self.broken = broken

    def probabilities(self, rows: Any) -> Any:
        if self.broken:
            raise RuntimeError("doesn't score")
        return np.full(len(rows), 0.25)


class FakeSource:
    """A registry in memory: a test changes the aliases and the files behind them."""

    def __init__(self, tmp_path: Path) -> None:
        self.store = tmp_path / "registry"
        self.refs: list[ModelRef] = []
        self.fetches: list[str] = []
        self.down = False

    def publish(self, alias: str, name: str, version: str, marker: str = "x") -> ModelRef:
        digest = make_artifact(self.store / f"{name}-{version}", marker)
        ref = ModelRef(alias, name, version, digest)
        # Moving an alias keeps its place in the chain, as it does in the registry.
        known = [item.alias for item in self.refs]
        if alias in known:
            self.refs[known.index(alias)] = ref
        else:
            self.refs.append(ref)
        return ref

    def resolve(self) -> list[ModelRef]:
        if self.down:
            raise httpx.ConnectError("registry down")
        return list(self.refs)

    def fetch(self, ref: ModelRef, target: Path) -> None:
        self.fetches.append(ref.model_name)
        source = self.store / f"{ref.model_name}-{ref.version}"
        for path in source.iterdir():
            (target / path.name).write_bytes(path.read_bytes())


def loader_for(broken: set[str] | None = None) -> Callable[[Path, str], Scorer]:
    def load(folder: Path, _digest: str) -> Scorer:
        return FakeModel(folder.name, any(name in folder.name for name in (broken or set())))

    return load


@pytest.fixture
def source(tmp_path: Path) -> FakeSource:
    fake = FakeSource(tmp_path)
    fake.publish("champion", "roberta", "1", "r1")
    fake.publish("fallback-1", "bilstm", "2", "b2")
    fake.publish("fallback-2", "mlp", "3", "m3")
    return fake


def chain_for(
    source: FakeSource, tmp_path: Path, broken: set[str] | None = None, **options: Any
) -> ServingChain:
    return ServingChain(
        source,
        ModelCache(tmp_path / "cache"),
        FAST_TIMEOUT_MS,
        loader=loader_for(broken),
        **options,
    )


def answered_by(chain: ServingChain) -> str:
    return chain.current.predictor.score([{"comment": "x"}]).model_name


# --- The verified cache ---


def test_a_download_is_cached_and_not_repeated(source: FakeSource, tmp_path: Path) -> None:
    cache = ModelCache(tmp_path / "cache")
    ref = source.refs[0]
    first = cache.ensure(ref, source)
    second = cache.ensure(ref, source)
    assert first == second
    assert source.fetches == ["roberta"]
    assert (first / "model.onnx").is_file()


def test_a_corrupt_cached_file_is_replaced(source: FakeSource, tmp_path: Path) -> None:
    cache = ModelCache(tmp_path / "cache")
    ref = source.refs[0]
    folder = cache.ensure(ref, source)
    (folder / "model.onnx").write_bytes(b"tampered")
    repaired = cache.ensure(ref, source)
    assert source.fetches == ["roberta", "roberta"]
    assert (repaired / "model.onnx").read_bytes() == b"r1" * 100


def test_an_extra_file_in_the_cache_is_treated_as_tampering(
    source: FakeSource, tmp_path: Path
) -> None:
    cache = ModelCache(tmp_path / "cache")
    ref = source.refs[0]
    folder = cache.ensure(ref, source)
    (folder / "extra.json").write_text("{}", "utf-8")
    assert not (cache.ensure(ref, source) / "extra.json").exists()
    assert source.fetches == ["roberta", "roberta"]


def test_a_corrupt_download_never_enters_the_cache(source: FakeSource, tmp_path: Path) -> None:
    cache = ModelCache(tmp_path / "cache")
    ref = source.refs[0]
    (source.store / "roberta-1" / "model.onnx").write_bytes(b"corrupted in the registry")
    with pytest.raises(ArtifactError):
        cache.ensure(ref, source)
    assert list((tmp_path / "cache").iterdir()) == []


def test_a_wrong_digest_is_refused_even_if_the_files_agree(
    source: FakeSource, tmp_path: Path
) -> None:
    good = source.refs[0]
    wrong = ModelRef(good.alias, good.model_name, good.version, "0" * 64)
    with pytest.raises(ArtifactError, match="registered digest"):
        ModelCache(tmp_path / "cache").ensure(wrong, source)


def test_prune_removes_old_versions_and_interrupted_downloads(
    source: FakeSource, tmp_path: Path
) -> None:
    cache = ModelCache(tmp_path / "cache")
    keep, drop = source.refs[0], source.refs[1]
    cache.ensure(keep, source)
    cache.ensure(drop, source)
    (tmp_path / "cache" / ".incoming-abc").mkdir()
    cache.prune([keep])
    assert sorted(path.name for path in (tmp_path / "cache").iterdir()) == [keep.folder_name]


# --- First load ---


def test_load_builds_the_chain_in_alias_order(source: FakeSource, tmp_path: Path) -> None:
    chain = chain_for(source, tmp_path)
    assert not chain.ready
    chain.load()
    assert chain.ready
    assert [item.model_name for item in chain.current.predictor.status()] == [
        "roberta",
        "bilstm",
        "mlp",
    ]
    assert answered_by(chain) == "roberta"


def test_a_broken_model_at_startup_shortens_the_chain(source: FakeSource, tmp_path: Path) -> None:
    chain = chain_for(source, tmp_path, broken={"roberta"})
    chain.load()
    assert answered_by(chain) == "bilstm"
    assert chain.last_error == "champion: AllModelsFailed"


def test_a_corrupt_artifact_at_startup_shortens_the_chain(
    source: FakeSource, tmp_path: Path
) -> None:
    (source.store / "roberta-1" / "model.onnx").write_bytes(b"corrupt")
    chain = chain_for(source, tmp_path)
    chain.load()
    assert answered_by(chain) == "bilstm"


def test_no_usable_model_at_all_is_an_error(source: FakeSource, tmp_path: Path) -> None:
    chain = chain_for(source, tmp_path, broken={"roberta", "bilstm", "mlp"})
    with pytest.raises(ChainError, match="could be loaded"):
        chain.load()
    assert not chain.ready


def test_render_mode_loads_only_the_champion_and_the_last_model(
    source: FakeSource, tmp_path: Path
) -> None:
    chain = chain_for(source, tmp_path, mode=ChainMode.CHAMPION_AND_LAST)
    chain.load()
    assert [item.model_name for item in chain.current.predictor.status()] == ["roberta", "mlp"]
    assert source.fetches == ["roberta", "mlp"]


def test_force_model_outside_the_loaded_chain_stops_startup(
    source: FakeSource, tmp_path: Path
) -> None:
    chain = chain_for(source, tmp_path, mode=ChainMode.CHAMPION_AND_LAST, force_model="bilstm")
    with pytest.raises(ValueError, match="FORCE_MODEL=bilstm"):
        chain.load()


def test_the_fast_predictor_has_no_transformers(source: FakeSource, tmp_path: Path) -> None:
    chain = chain_for(source, tmp_path)
    chain.load()
    assert [item.model_name for item in chain.current.fast.status()] == ["bilstm", "mlp"]


# --- Registry outage ---


def test_a_restart_during_a_registry_outage_uses_the_saved_chain(
    source: FakeSource, tmp_path: Path
) -> None:
    chain_for(source, tmp_path).load()
    fetched = list(source.fetches)
    source.down = True
    restarted = chain_for(source, tmp_path)
    restarted.load()
    assert answered_by(restarted) == "roberta"
    assert source.fetches == fetched


def test_no_registry_and_no_saved_chain_is_an_error(source: FakeSource, tmp_path: Path) -> None:
    source.down = True
    with pytest.raises(ChainError, match="no saved chain"):
        chain_for(source, tmp_path).load()


def test_a_shortened_chain_isnt_saved_as_the_outage_fallback(
    source: FakeSource, tmp_path: Path
) -> None:
    chain_for(source, tmp_path, broken={"roberta"}).load()
    assert ModelCache(tmp_path / "cache").load_manifest() == []


# --- Reload ---


def test_reload_without_a_change_does_nothing(source: FakeSource, tmp_path: Path) -> None:
    chain = chain_for(source, tmp_path)
    chain.load()
    before = chain.current
    assert chain.reload() is False
    assert chain.current is before


def test_reload_swaps_when_an_alias_moves(source: FakeSource, tmp_path: Path) -> None:
    changes: list[int] = []
    chain = chain_for(source, tmp_path, on_change=lambda: changes.append(1))
    chain.load()
    source.publish("champion", "distilbert", "9", "d9")
    assert chain.reload() is True
    assert answered_by(chain) == "distilbert"
    assert changes == [1]


def test_reload_reuses_models_already_in_memory(source: FakeSource, tmp_path: Path) -> None:
    chain = chain_for(source, tmp_path)
    chain.load()
    kept = chain.current.models[("mlp", "3")]
    source.publish("champion", "distilbert", "9", "d9")
    chain.reload()
    assert chain.current.models[("mlp", "3")] is kept
    assert source.fetches.count("mlp") == 1


def test_reload_keeps_the_old_chain_when_a_new_model_is_corrupt(
    source: FakeSource, tmp_path: Path
) -> None:
    chain = chain_for(source, tmp_path)
    chain.load()
    before = chain.current
    source.publish("champion", "distilbert", "9", "d9")
    (source.store / "distilbert-9" / "model.onnx").write_bytes(b"corrupt")
    assert chain.reload() is False
    assert chain.current is before
    assert answered_by(chain) == "roberta"
    assert chain.reload_failures == 1
    assert chain.last_error == "ArtifactError"


def test_reload_keeps_the_old_chain_when_a_new_model_fails_its_smoke_test(
    source: FakeSource, tmp_path: Path
) -> None:
    chain = chain_for(source, tmp_path, broken={"distilbert"})
    chain.load()
    source.publish("champion", "distilbert", "9", "d9")
    assert chain.reload() is False
    assert answered_by(chain) == "roberta"
    assert chain.last_error == "AllModelsFailed"


def test_reload_during_a_registry_outage_keeps_serving(source: FakeSource, tmp_path: Path) -> None:
    chain = chain_for(source, tmp_path)
    chain.load()
    source.down = True
    assert chain.reload() is False
    assert answered_by(chain) == "roberta"
    assert chain.last_error == "ConnectError"


def test_reload_recovers_after_a_failure(source: FakeSource, tmp_path: Path) -> None:
    chain = chain_for(source, tmp_path)
    chain.load()
    source.down = True
    chain.reload()
    source.down = False
    assert chain.reload() is False
    assert chain.last_error is None


def test_reload_removes_versions_no_longer_in_use(source: FakeSource, tmp_path: Path) -> None:
    chain = chain_for(source, tmp_path)
    chain.load()
    old = source.refs[0].folder_name
    source.publish("champion", "distilbert", "9", "d9")
    chain.reload()
    assert not (tmp_path / "cache" / old).exists()


def test_reads_during_a_swap_always_see_a_whole_chain(source: FakeSource, tmp_path: Path) -> None:
    chain = chain_for(source, tmp_path)
    chain.load()
    source.publish("champion", "distilbert", "9", "d9")
    seen: set[str] = set()
    stop = threading.Event()

    def read() -> None:
        while not stop.is_set():
            seen.add(answered_by(chain))

    thread = threading.Thread(target=read)
    thread.start()
    chain.reload()
    stop.set()
    thread.join()
    assert seen <= {"roberta", "distilbert"}
    assert answered_by(chain) == "distilbert"


def test_polling_picks_up_an_alias_change_and_stops(source: FakeSource, tmp_path: Path) -> None:
    swapped = threading.Event()
    chain = chain_for(source, tmp_path, on_change=swapped.set)
    chain.load()
    source.publish("champion", "distilbert", "9", "d9")
    stop = threading.Event()
    thread = chain.poll(0.01, stop)
    assert swapped.wait(10)
    stop.set()
    thread.join(5)
    assert not thread.is_alive()


# --- The MLflow REST client, against a recorded shape ---

RUN = "e3334f8397ad4830a6821cb33f09f0d9"


def mlflow_app(
    tmp_path: Path, listing: list[dict[str, Any]] | None = None
) -> Callable[[httpx.Request], httpx.Response]:
    """Answers as DagsHub's MLflow did on 2026-11-07, for the same three endpoints."""
    artifact = tmp_path / "published"
    digest = make_artifact(artifact, "m")

    def handle(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/registered-models/alias"):
            alias = request.url.params["alias"]
            if alias != "champion":
                return httpx.Response(400, json={"error_code": "INVALID_PARAMETER_VALUE"})
            return httpx.Response(
                200,
                json={
                    "model_version": {
                        "name": "wikiguard",
                        "version": "5",
                        "run_id": RUN,
                        "status": "READY",
                        "tags": [
                            {"key": "artifact_sha256", "value": digest},
                            {"key": "model_name", "value": "mlp"},
                        ],
                    }
                },
            )
        if request.url.path.endswith("/artifacts/list"):
            files = listing or [
                {"path": f"model/{path.name}", "is_dir": False} for path in artifact.iterdir()
            ]
            return httpx.Response(200, json={"files": files})
        if request.url.path.endswith("/get-artifact"):
            name = request.url.params["path"].removeprefix("model/")
            return httpx.Response(200, content=(artifact / name).read_bytes())
        return httpx.Response(500)

    return handle


def test_registry_resolves_until_the_first_missing_alias(tmp_path: Path) -> None:
    registry = MlflowRegistry(
        "https://registry.test/x.mlflow", transport=httpx.MockTransport(mlflow_app(tmp_path))
    )
    refs = registry.resolve()
    assert [(ref.alias, ref.model_name, ref.version, ref.run_id) for ref in refs] == [
        ("champion", "mlp", "5", RUN)
    ]


def test_registry_download_passes_the_cache_check(tmp_path: Path) -> None:
    registry = MlflowRegistry(
        "https://registry.test/x.mlflow", transport=httpx.MockTransport(mlflow_app(tmp_path))
    )
    ref = registry.resolve()[0]
    folder = ModelCache(tmp_path / "cache").ensure(ref, registry)
    assert (folder / "model.onnx").read_bytes() == b"m" * 100


@pytest.mark.parametrize(
    "path",
    ["model/../../evil.json", "model/..", "model/sub/file.json", "model/notes.pkl"],
    ids=["escape", "dots", "nested", "pickle"],
)
def test_a_hostile_artifact_listing_is_refused(tmp_path: Path, path: str) -> None:
    listing = [{"path": path, "is_dir": False}]
    registry = MlflowRegistry(
        "https://registry.test/x.mlflow",
        transport=httpx.MockTransport(mlflow_app(tmp_path, listing)),
    )
    ref = registry.resolve()[0]
    with pytest.raises(ArtifactError):
        ModelCache(tmp_path / "cache").ensure(ref, registry)
    assert not (tmp_path / "evil.json").exists()


def test_a_standard_404_for_a_missing_alias_also_ends_the_chain() -> None:
    def handle(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={"error_code": "RESOURCE_DOES_NOT_EXIST"})

    registry = MlflowRegistry(
        "https://registry.test/x.mlflow", transport=httpx.MockTransport(handle)
    )
    with pytest.raises(ChainError, match="no champion"):
        registry.resolve()


@pytest.mark.parametrize("status", [401, 403, 500, 503])
def test_a_registry_failure_isnt_mistaken_for_the_end_of_the_chain(status: int) -> None:
    registry = MlflowRegistry(
        "https://registry.test/x.mlflow",
        transport=httpx.MockTransport(lambda _: httpx.Response(status, json={})),
    )
    with pytest.raises(httpx.HTTPStatusError):
        registry.resolve()


def test_a_registry_with_no_champion_is_an_error() -> None:
    registry = MlflowRegistry(
        "https://registry.test/x.mlflow",
        transport=httpx.MockTransport(lambda _: httpx.Response(404, json={})),
    )
    with pytest.raises(ChainError, match="no champion"):
        registry.resolve()


# --- Local runs and settings ---


def test_local_runs_reads_the_trainers_aliases(tmp_path: Path) -> None:
    digest = make_artifact(tmp_path / "runs" / "mlp" / "model")
    (tmp_path / "runs" / "aliases.json").write_text(
        json.dumps({"champion": {"version": 5, "model": "mlp"}}), "utf-8"
    )
    source = LocalRuns(tmp_path / "runs")
    refs = source.resolve()
    assert refs == [ModelRef("champion", "mlp", "5", digest)]
    assert (ModelCache(tmp_path / "cache").ensure(refs[0], source) / "config.json").is_file()


def test_the_registry_address_is_required_for_the_mlflow_source(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MODEL_SOURCE", "mlflow")
    settings = ServingSettings(_env_file=None)  # type: ignore[call-arg]
    with pytest.raises(ChainError, match="MLFLOW_TRACKING_URI"):
        source_from_settings(settings)
