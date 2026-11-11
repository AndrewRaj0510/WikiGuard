"""Load the serving chain by registry alias, keep it on disk, and swap it safely when aliases move.

No MLflow client here. Serving images talk to the registry over plain HTTP, because the client's
dependencies would take space on a 512 MB service and MLflow must stay out of serving (F-049).

Rules that aren't obvious:

- A model is checked against its registered digest before anything loads it, both after a download
  and on every load from disk. A bad cached folder is deleted and fetched again.
- A download lands in a temporary folder and is renamed only after it passes, so a crash can't
  leave a half-written folder that looks like a model.
- The first load tolerates a bad model and serves the rest of the chain, because some answer beats
  none. A reload is all or nothing: on any failure the old chain keeps serving.
- The last good chain is saved beside the models, so a registry outage can't stop a restart.
"""

import json
import logging
import os
import shutil
import threading
import uuid
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any, Protocol

import httpx

from wikiguard_core.artifacts import (
    ALLOWED_SUFFIXES,
    CHECKSUMS_FILE,
    ArtifactError,
    sha256_file,
    verify_artifact,
)
from wikiguard_core.features import build_features
from wikiguard_core.predictor import ChainEntry, Predictor, Scorer
from wikiguard_core.schemas import EditEvent

logger = logging.getLogger(__name__)

REGISTERED_MODEL = "wikiguard"
DIGEST_TAG = "artifact_sha256"
NAME_TAG = "model_name"
ARTIFACT_FOLDER = "model"
MAX_FALLBACKS = 9
CHAIN_ALIASES = ("champion", *(f"fallback-{index}" for index in range(1, MAX_FALLBACKS + 1)))
# The model in its shadow period: scored on live edits, never served (PLAN.md, section 4).
CANDIDATE_ALIAS = "candidate"
MISSING_ALIAS_CODES = {"INVALID_PARAMETER_VALUE", "RESOURCE_DOES_NOT_EXIST"}
HTTP_TIMEOUT_SECONDS = 60.0
MAX_ARTIFACT_BYTES = 512 * 1024 * 1024
DOWNLOAD_CHUNK_BYTES = 1024 * 1024
MANIFEST_FILE = "chain.json"
SMOKE_TIMEOUT_MS = 60_000
ALIASES_FILE = "aliases.json"
# Models fast enough to run when the scorer is behind. Decided by architecture, not rank:
# transformers take tens of milliseconds an edit, and these take under one.
FAST_MODELS = frozenset({"bilstm", "mlp"})
SMOKE_EDIT = EditEvent(
    rev_id=1,
    parent_rev_id=None,
    title="Smoke test",
    timestamp=datetime(2026, 1, 1, tzinfo=UTC),
    is_minor=False,
    is_anonymous=False,
    user_hash=None,
    comment="smoke test",
    old_length=10,
    new_length=20,
    added_text="a few added words",
    removed_text="a removed word",
)


class ChainMode(StrEnum):
    FULL = "full"
    # Render has 512 MB: the champion and the last-resort MLP, nothing between (DEC-050).
    CHAMPION_AND_LAST = "champion_and_last"


class ChainError(RuntimeError):
    """No usable chain could be built."""


@dataclass(frozen=True)
class ModelRef:
    alias: str
    model_name: str
    version: str
    artifact_sha256: str
    run_id: str = ""

    @property
    def folder_name(self) -> str:
        return f"{self.model_name}-{self.version}-{self.artifact_sha256[:12]}"


class ModelSource(Protocol):
    def resolve(self) -> list[ModelRef]: ...

    def resolve_alias(self, alias: str) -> ModelRef | None: ...

    def fetch(self, ref: ModelRef, target: Path) -> None: ...


Loader = Callable[[Path, str], Scorer]


def _error_code(response: httpx.Response) -> str | None:
    try:
        code: str | None = response.json().get("error_code")
    except ValueError:
        return None
    return code


class MlflowRegistry:
    """Reads aliases and downloads artifacts over MLflow's REST API. It changes nothing."""

    def __init__(
        self,
        tracking_uri: str,
        username: str | None = None,
        password: str | None = None,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        auth = httpx.BasicAuth(username, password) if username and password else None
        self._client = httpx.Client(
            base_url=tracking_uri.rstrip("/"),
            auth=auth,
            timeout=HTTP_TIMEOUT_SECONDS,
            follow_redirects=True,
            transport=transport,
        )

    def _version_for(self, alias: str) -> dict[str, Any] | None:
        response = self._client.get(
            "/api/2.0/mlflow/registered-models/alias",
            params={"name": REGISTERED_MODEL, "alias": alias},
        )
        if response.status_code == httpx.codes.NOT_FOUND:
            return None
        # DagsHub's server answers 400 INVALID_PARAMETER_VALUE for an alias it doesn't have (F-099).
        if response.status_code == httpx.codes.BAD_REQUEST and (
            _error_code(response) in MISSING_ALIAS_CODES
        ):
            return None
        response.raise_for_status()
        version: dict[str, Any] = response.json()["model_version"]
        return version

    def resolve_alias(self, alias: str) -> ModelRef | None:
        version = self._version_for(alias)
        if version is None:
            return None
        tags = {tag["key"]: tag["value"] for tag in version.get("tags", [])}
        if DIGEST_TAG not in tags or NAME_TAG not in tags:
            raise ChainError(f"{alias} has no recorded checksum or model name")
        return ModelRef(
            alias, tags[NAME_TAG], str(version["version"]), tags[DIGEST_TAG], version["run_id"]
        )

    def resolve(self) -> list[ModelRef]:
        refs: list[ModelRef] = []
        for alias in CHAIN_ALIASES:
            ref = self.resolve_alias(alias)
            if ref is None:
                break
            refs.append(ref)
        if not refs:
            raise ChainError("the registry has no champion")
        return refs

    def fetch(self, ref: ModelRef, target: Path) -> None:
        response = self._client.get(
            "/api/2.0/mlflow/artifacts/list", params={"run_id": ref.run_id, "path": ARTIFACT_FOLDER}
        )
        response.raise_for_status()
        for item in response.json().get("files", []):
            path = str(item["path"])
            name = path.removeprefix(f"{ARTIFACT_FOLDER}/")
            # The listing comes from the network, so it never gets to choose where a file lands.
            if "/" in name or "\\" in name or name in {"", ".", ".."} or item.get("is_dir"):
                raise ArtifactError(f"unexpected entry in the artifact listing: {path!r}")
            if Path(name).suffix not in ALLOWED_SUFFIXES:
                raise ArtifactError(f"{name} isn't an allowed artifact format")
            self._download(ref.run_id, path, target / name)

    def _download(self, run_id: str, path: str, target: Path) -> None:
        written = 0
        with self._client.stream(
            "GET", "/get-artifact", params={"path": path, "run_uuid": run_id}
        ) as response:
            response.raise_for_status()
            with target.open("wb") as file:
                for chunk in response.iter_bytes(DOWNLOAD_CHUNK_BYTES):
                    written += len(chunk)
                    if written > MAX_ARTIFACT_BYTES:
                        raise ArtifactError(f"{target.name} is larger than the size limit")
                    file.write(chunk)


class LocalRuns:
    """The trainer's output folder, for Compose and tests, where no registry is needed.

    aliases.json maps each alias to a model and version. An entry may name its own folder, relative
    to the root, which a local registry with two versions of one model needs.
    """

    def __init__(self, root: Path) -> None:
        self.root = root

    def _aliases(self) -> dict[str, dict[str, Any]]:
        aliases: dict[str, dict[str, Any]] = json.loads(
            (self.root / ALIASES_FILE).read_text("utf-8")
        )
        return aliases

    def _folder(self, entry: dict[str, Any]) -> Path:
        return self.root / str(entry.get("folder", entry["model"])) / ARTIFACT_FOLDER

    def _ref(self, alias: str, entry: dict[str, Any]) -> ModelRef:
        return ModelRef(
            alias,
            entry["model"],
            str(entry["version"]),
            sha256_file(self._folder(entry) / CHECKSUMS_FILE),
            entry.get("folder", entry["model"]),
        )

    def resolve_alias(self, alias: str) -> ModelRef | None:
        entry = self._aliases().get(alias)
        return None if entry is None else self._ref(alias, entry)

    def resolve(self) -> list[ModelRef]:
        aliases = self._aliases()
        refs = []
        # Serving order, whatever order the file lists them in. Other aliases, such as the
        # candidate in shadow, aren't part of the chain.
        for alias in CHAIN_ALIASES:
            if alias not in aliases:
                break
            refs.append(self._ref(alias, aliases[alias]))
        return refs

    def fetch(self, ref: ModelRef, target: Path) -> None:
        # run_id holds the folder for a local source.
        folder = self.root / (ref.run_id or ref.model_name) / ARTIFACT_FOLDER
        shutil.copytree(folder, target, dirs_exist_ok=True)


class ModelCache:
    """Model folders on local disk, named by version and digest, each checked before use."""

    def __init__(self, root: Path) -> None:
        self.root = root

    def _valid(self, folder: Path, digest: str) -> bool:
        if not folder.is_dir():
            return False
        try:
            verify_artifact(folder, digest)
        except (ArtifactError, OSError, ValueError, KeyError) as error:
            logger.warning("Cached %s is bad, fetching it again: %s", folder.name, error)
            shutil.rmtree(folder, ignore_errors=True)
            return False
        return True

    def ensure(self, ref: ModelRef, source: ModelSource) -> Path:
        folder = self.root / ref.folder_name
        if self._valid(folder, ref.artifact_sha256):
            return folder
        self.root.mkdir(parents=True, exist_ok=True)
        temporary = self.root / f".incoming-{uuid.uuid4().hex}"
        temporary.mkdir()
        try:
            source.fetch(ref, temporary)
            verify_artifact(temporary, ref.artifact_sha256)
            os.replace(temporary, folder)
        finally:
            shutil.rmtree(temporary, ignore_errors=True)
        return folder

    def save_manifest(self, refs: Sequence[ModelRef]) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        temporary = self.root / f"{MANIFEST_FILE}.{uuid.uuid4().hex}"
        temporary.write_text(json.dumps([asdict(ref) for ref in refs], indent=2), "utf-8")
        os.replace(temporary, self.root / MANIFEST_FILE)

    def load_manifest(self) -> list[ModelRef]:
        path = self.root / MANIFEST_FILE
        if not path.is_file():
            return []
        return [ModelRef(**entry) for entry in json.loads(path.read_text("utf-8"))]

    def prune(self, keep: Sequence[ModelRef]) -> None:
        """Remove versions no longer in the chain, and anything a crash left half-written."""
        wanted = {ref.folder_name for ref in keep}
        if not self.root.is_dir():
            return
        for path in self.root.iterdir():
            if path.is_dir() and path.name not in wanted:
                shutil.rmtree(path, ignore_errors=True)


def load_for_serving(threads: int = 1) -> Loader:
    """The real loader, imported late so this module needs no ONNX Runtime to be tested."""

    def load(folder: Path, digest: str) -> Scorer:
        from wikiguard_core.inference import OneAtATime, ServedModel

        return OneAtATime(ServedModel(folder, threads=threads, expected_digest=digest))

    return load


def smoke_check(alias: str, name: str, version: str, model: Scorer) -> None:
    """A model must score one edit before it serves. On a slot, so a hang can't stop a reload."""
    row = build_features(SMOKE_EDIT).as_dict()
    Predictor([ChainEntry(alias, name, version, model)], SMOKE_TIMEOUT_MS).score([row])


@dataclass(frozen=True)
class LoadedChain:
    refs: tuple[ModelRef, ...]
    predictor: Predictor
    fast: Predictor
    models: dict[tuple[str, str], Scorer]


class ServingChain:
    def __init__(
        self,
        source: ModelSource,
        cache: ModelCache,
        timeout_ms: int,
        force_model: str | None = None,
        mode: ChainMode = ChainMode.FULL,
        loader: Loader | None = None,
        on_change: Callable[[], None] | None = None,
    ) -> None:
        self._source = source
        self._cache = cache
        self._timeout_ms = timeout_ms
        self._force_model = force_model
        self._mode = mode
        self._loader = loader or load_for_serving()
        self._on_change = on_change
        self._loaded: LoadedChain | None = None
        self._reload_lock = threading.Lock()
        self.last_error: str | None = None
        self.reload_failures = 0

    @property
    def ready(self) -> bool:
        return self._loaded is not None

    @property
    def current(self) -> LoadedChain:
        loaded = self._loaded
        if loaded is None:
            raise ChainError("the chain hasn't loaded yet")
        return loaded

    def _resolve(self) -> list[ModelRef]:
        refs = self._source.resolve()
        if self._mode is ChainMode.CHAMPION_AND_LAST and len(refs) > 1:
            refs = [refs[0], refs[-1]]
        return refs

    def _build(
        self, refs: Sequence[ModelRef], models: dict[tuple[str, str], Scorer]
    ) -> LoadedChain:
        entries = [
            ChainEntry(
                ref.alias, ref.model_name, ref.version, models[(ref.model_name, ref.version)]
            )
            for ref in refs
        ]
        fast = [entry for entry in entries if entry.model_name in FAST_MODELS] or entries
        return LoadedChain(
            tuple(refs),
            Predictor(entries, self._timeout_ms, force_model=self._force_model),
            Predictor(fast, self._timeout_ms, force_model=self._force_model),
            models,
        )

    def _load_model(self, ref: ModelRef) -> Scorer:
        folder = self._cache.ensure(ref, self._source)
        model = self._loader(folder, ref.artifact_sha256)
        smoke_check(ref.alias, ref.model_name, ref.version, model)
        return model

    def load(self) -> None:
        """Startup. A model that won't load is left out, and the rest of the chain serves."""
        try:
            refs = self._resolve()
        except (httpx.HTTPError, ChainError, ValueError, KeyError, OSError) as error:
            logger.warning("Registry unavailable, using the saved chain: %s", type(error).__name__)
            refs = self._cache.load_manifest()
            if self._mode is ChainMode.CHAMPION_AND_LAST and len(refs) > 1:
                refs = [refs[0], refs[-1]]
            if not refs:
                raise ChainError("no registry and no saved chain") from error
        models: dict[tuple[str, str], Scorer] = {}
        usable: list[ModelRef] = []
        for ref in refs:
            try:
                models[(ref.model_name, ref.version)] = self._load_model(ref)
                usable.append(ref)
            except Exception as error:  # noqa: BLE001 - any failure here just shortens the chain
                self.last_error = f"{ref.alias}: {type(error).__name__}"
                logger.error("Left %s out of the chain: %s", ref.alias, type(error).__name__)
        if not usable:
            raise ChainError("no model in the chain could be loaded")
        self._loaded = self._build(usable, models)
        # Saved only when complete, so the fallback after an outage is never a shortened chain.
        if len(usable) == len(refs):
            self._cache.save_manifest(refs)
        logger.info("Chain loaded: %s", [ref.folder_name for ref in usable])

    def reload(self) -> bool:
        """Swap to the registry's current chain if it changed. Returns True if it swapped.

        The new chain is complete and tested before one reference is replaced, so a request sees
        the old chain or the new one, never a mix. In-flight calls finish on the old models.
        """
        with self._reload_lock:
            try:
                refs = self._resolve()
                current = self._loaded
                if current is not None and tuple(refs) == current.refs:
                    self.last_error = None
                    return False
                known = current.models if current is not None else {}
                models = {
                    (ref.model_name, ref.version): known.get((ref.model_name, ref.version))
                    or self._load_model(ref)
                    for ref in refs
                }
                loaded = self._build(refs, models)
            except Exception as error:  # noqa: BLE001 - whatever went wrong, the old chain stays
                self.reload_failures += 1
                self.last_error = type(error).__name__
                logger.error("Reload failed, keeping the current chain: %s", type(error).__name__)
                return False
            self._loaded = loaded
            self.last_error = None
            self._cache.save_manifest(refs)
            self._cache.prune(refs)
            logger.info("Chain swapped: %s", [ref.folder_name for ref in refs])
        if self._on_change is not None:
            self._on_change()
        return True

    def poll(self, interval_seconds: float, stop: threading.Event) -> threading.Thread:
        """Check for alias changes on a daemon thread. Stops when the event is set."""

        def run() -> None:
            while not stop.wait(interval_seconds):
                self.reload()

        thread = threading.Thread(target=run, name="alias-poll", daemon=True)
        thread.start()
        return thread


class ShadowSlot:
    """The candidate in its shadow period, loaded beside the chain and never part of it.

    It has its own cache folder, because a chain reload prunes every folder it doesn't use. A
    candidate that won't load just means no shadow scores, which the shadow check counts as a
    failure: it never touches the chain that serves users.
    """

    def __init__(
        self,
        source: ModelSource,
        cache: ModelCache,
        timeout_ms: int,
        loader: Loader | None = None,
    ) -> None:
        self._source = source
        self._cache = cache
        self._timeout_ms = timeout_ms
        self._loader = loader or load_for_serving()
        self._current: tuple[ModelRef, Predictor] | None = None
        self._lock = threading.Lock()
        self.last_error: str | None = None

    @property
    def current(self) -> tuple[ModelRef, Predictor] | None:
        return self._current

    def refresh(self) -> bool:
        """Follow the candidate alias. Returns True if what's loaded changed."""
        with self._lock:
            try:
                ref = self._source.resolve_alias(CANDIDATE_ALIAS)
            except Exception as error:  # noqa: BLE001 - keep whatever is loaded
                self.last_error = type(error).__name__
                return False
            current = self._current
            if ref is None:
                self._current = None
                self.last_error = None
                return current is not None
            if current is not None and current[0] == ref:
                return False
            try:
                folder = self._cache.ensure(ref, self._source)
                model = self._loader(folder, ref.artifact_sha256)
                smoke_check(ref.alias, ref.model_name, ref.version, model)
            except Exception as error:  # noqa: BLE001 - a bad candidate gets no shadow scores
                self._current = None
                self.last_error = f"{ref.folder_name}: {type(error).__name__}"
                logger.error("Shadow candidate didn't load: %s", self.last_error)
                return current is not None
            entry = ChainEntry(ref.alias, ref.model_name, ref.version, model)
            # Nothing behind it, so a failure is reported as one, never covered by a fallback.
            self._current = (ref, Predictor([entry], self._timeout_ms))
            self._cache.prune([ref])
            self.last_error = None
            logger.info("Shadow candidate loaded: %s", ref.folder_name)
            return True

    def poll(self, interval_seconds: float, stop: threading.Event) -> threading.Thread:
        def run() -> None:
            while not stop.wait(interval_seconds):
                self.refresh()

        thread = threading.Thread(target=run, name="shadow-poll", daemon=True)
        thread.start()
        return thread


def shadow_from_settings(settings: Any, loader: Loader | None = None) -> ShadowSlot:
    return ShadowSlot(
        source_from_settings(settings),
        ModelCache(Path(f"{settings.model_cache_dir}-shadow")),
        settings.predict_timeout_ms,
        loader or load_for_serving(settings.onnx_threads),
    )


def source_from_settings(settings: Any) -> ModelSource:
    if settings.model_source == "local":
        return LocalRuns(Path(settings.local_runs_path))
    if not settings.mlflow_tracking_uri:
        raise ChainError("MLFLOW_TRACKING_URI is required when MODEL_SOURCE is mlflow")
    password = settings.mlflow_tracking_password
    return MlflowRegistry(
        settings.mlflow_tracking_uri,
        settings.mlflow_tracking_username,
        password.get_secret_value() if password else None,
    )


def chain_from_settings(
    settings: Any, on_change: Callable[[], None] | None = None, loader: Loader | None = None
) -> ServingChain:
    return ServingChain(
        source_from_settings(settings),
        ModelCache(Path(settings.model_cache_dir)),
        settings.predict_timeout_ms,
        settings.force_model,
        ChainMode(settings.model_chain_mode),
        loader or load_for_serving(settings.onnx_threads),
        on_change,
    )
