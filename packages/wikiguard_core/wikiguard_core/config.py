"""Validated settings, read from the environment or a local .env file.

Settings are split by area so each service asks only for what it uses. The ingestor can then
start without the LLM key, and a leak in one service exposes fewer secrets. A service combines
the areas it needs and calls load_settings() once at startup:

    class ScorerSettings(CommonSettings, DatabaseSettings, KafkaSettings, ServingSettings):
        pass

    settings = load_settings(ScorerSettings)
"""

from typing import Literal, TypeVar

from pydantic import AliasChoices, Field, SecretStr, ValidationError, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

MIN_SECRET_KEY_LENGTH = 32


class ConfigError(RuntimeError):
    """The configuration is missing or invalid. The message never includes values."""


class BaseConfig(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        # A blank line in .env counts as unset, so a blank secret fails instead of passing.
        env_ignore_empty=True,
        # .env holds settings for every service, and each service reads only its own.
        extra="ignore",
        frozen=True,
        hide_input_in_errors=True,
    )


class CommonSettings(BaseConfig):
    environment: Literal["dev", "staging", "prod"] = "dev"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"


class DatabaseSettings(BaseConfig):
    database_url: SecretStr


class MigrationSettings(BaseConfig):
    # Separate because this login can change the schema, so only the migration job should hold it.
    database_migrations_url: SecretStr


class KafkaSettings(BaseConfig):
    kafka_bootstrap_servers: str = "redpanda:9092"


class WikimediaSettings(BaseConfig):
    wikimedia_user_agent: str
    wikimedia_stream_url: str = "https://stream.wikimedia.org/v2/stream/recentchange"
    mediawiki_api_url: str = "https://en.wikipedia.org/w/api.php"


class PrivacySettings(BaseConfig):
    user_hash_key: SecretStr = Field(min_length=MIN_SECRET_KEY_LENGTH)


class MlflowSettings(BaseConfig):
    mlflow_tracking_uri: str
    mlflow_tracking_username: str
    mlflow_tracking_password: SecretStr


class ServingSettings(BaseConfig):
    force_model: str | None = None
    predict_timeout_ms: int = Field(default=300, gt=0)
    # mlflow reads the registry. local reads the trainer's folder, for Compose and tests.
    model_source: Literal["mlflow", "local"] = "mlflow"
    model_cache_dir: str = "data/model_cache"
    local_runs_path: str = "data/runs"
    # 0 turns polling off. Render does, because a reload holds two models at once (DEC-055).
    model_poll_seconds: int = Field(default=300, ge=0)
    # champion_and_last on Render, where 512 MB can't hold the whole chain (DEC-050).
    model_chain_mode: Literal["full", "champion_and_last"] = "full"
    onnx_threads: int = Field(default=1, gt=0)
    # Optional: DagsHub's MLflow can be read without a login (F-093). Use a read-only token if set.
    mlflow_tracking_uri: str | None = None
    mlflow_tracking_username: str | None = None
    mlflow_tracking_password: SecretStr | None = None


class ScorerSettings(BaseConfig):
    scorer_batch_size: int = Field(default=32, gt=0)
    scorer_batch_window_ms: int = Field(default=50, ge=0)
    # Past this many edits behind, use the fast models. At half of it, go back (F-100).
    scorer_shed_lag: int = Field(default=500, gt=0)
    scorer_rescore_seconds: int = Field(default=60, gt=0)


class ApiSettings(BaseConfig):
    session_secret_key: SecretStr = Field(min_length=MIN_SECRET_KEY_LENGTH)
    user_password_hash: SecretStr
    metrics_bearer_token: SecretStr
    cors_allowed_origins: str = "http://localhost:3000"
    # The Worker sends this header. Set it and direct requests are refused, and only then is the
    # Worker's client-IP header believed. Blank in local dev.
    edge_shared_secret: SecretStr | None = None
    guest_signin_wait_seconds: int = Field(default=60, ge=0)
    predict_max_concurrency: int = Field(default=2, gt=0)
    db_statement_timeout_ms: int = Field(default=3000, gt=0)
    db_pool_size: int = Field(default=5, gt=0)
    max_body_bytes: int = Field(default=128 * 1024, gt=0)
    # The commit this build came from, shown by /health so a deploy can be told from the one before
    # it. Render sets RENDER_GIT_COMMIT itself. Blank locally. Hex only, so nothing else can be
    # echoed back to a caller.
    git_commit: str = Field(
        default="",
        pattern=r"^[0-9a-f]{0,40}$",
        validation_alias=AliasChoices("git_commit", "render_git_commit"),
    )


class FeatureFlagSettings(BaseConfig):
    explain_enabled: bool = True
    predict_enabled: bool = True
    read_only_mode: bool = False
    gate_enabled: bool = True


class GateSettings(BaseConfig):
    guest_idle_minutes: int = Field(default=5, gt=0)
    guest_max_minutes: int = Field(default=15, gt=0)
    gate_heartbeat_seconds: int = Field(default=15, gt=0)
    gate_lease_seconds: int = Field(default=45, gt=0)
    guest_explain_limit: int = Field(default=5, ge=0)


class LlmSettings(BaseConfig):
    llm_model: str
    llm_backup_model: str
    openai_api_key: SecretStr
    gemini_api_key: SecretStr
    explain_daily_call_cap: int = Field(gt=0)


class RetrainSettings(BaseConfig):
    retrain_interval_days: int = 7
    retrain_on_drift: bool = True
    retrain_startup_delay_seconds: int = Field(default=300, ge=0)
    # What the retrain-request poller runs, with the request's kind and id added as arguments.
    # Blank means requests are recorded as failed.
    trainer_command: str | None = None
    # The CPU models only: BERT and RoBERTa need a GPU notebook run by hand (DEC-022).
    retrain_models: str = "mlp,bilstm,distilbert"
    # The whole run stops starting new models past this, and the poller kills it at 12 hours.
    retrain_timeout_minutes: int = Field(default=600, gt=0)
    # Leaves the rest of the VM's CPUs to the scorer, which must keep up with the stream.
    retrain_threads: int = Field(default=2, gt=0)
    retrain_snapshot_rows: int = Field(default=100_000, ge=1_000)
    # How long a candidate scores live edits before the health gate. 24 in production; the
    # simulation uses 0 because it replays the period.
    shadow_hours: int = Field(default=24, ge=0)

    @field_validator("retrain_interval_days")
    @classmethod
    def check_supported_interval(cls, value: int) -> int:
        if value not in {1, 7, 14}:
            raise ValueError("must be 1, 7, or 14")
        return value

    @field_validator("retrain_models")
    @classmethod
    def check_cpu_models(cls, value: str) -> str:
        names = [name.strip() for name in value.split(",") if name.strip()]
        allowed = {"mlp", "bilstm", "distilbert"}
        if not names or not set(names) <= allowed:
            raise ValueError("must list only mlp, bilstm, and distilbert (DEC-022)")
        return ",".join(names)

    @property
    def model_names(self) -> list[str]:
        return self.retrain_models.split(",")


class StorageSettings(BaseConfig):
    # Where the DVC-tracked snapshots live: the repo checkout the job works in.
    snapshot_repo_path: str = "."
    # `dvc gc --cloud` deletes every version the checkout doesn't point at, locally too. Off unless
    # the job runs in its own clone, so it can't delete the laptop's copy of old data (F-093).
    snapshot_gc_enabled: bool = False
    # Warn when DagsHub's storage passes this. Its limit isn't confirmed (F-071), so it's low.
    storage_alert_bytes: int = Field(default=10 * 1024**3, gt=0)


class WorkerSettings(BaseConfig):
    # Workers serve /health and /metrics on this port.
    ops_port: int = Field(default=8000, gt=0)
    # Where the ingestor keeps its stream position. A mounted volume in the cluster.
    ingest_state_path: str = "data/ingestor_state.json"


class ArchiveSettings(BaseConfig):
    archive_path: str = "data/archive"
    # Edits under 48 hours old must never be deleted, so the window can't go below 3 days.
    archive_retention_days: int = Field(default=30, ge=3)
    archive_s3_endpoint: str | None = None
    archive_s3_bucket: str | None = None
    archive_s3_access_key: SecretStr | None = None
    archive_s3_secret_key: SecretStr | None = None


class AlertSettings(BaseConfig):
    sentry_dsn: SecretStr | None = None
    discord_webhook_url: SecretStr | None = None
    healthchecks_ping_url: SecretStr | None = None


SettingsT = TypeVar("SettingsT", bound=BaseConfig)


def load_settings(settings_class: type[SettingsT], env_file: str | None = ".env") -> SettingsT:
    """Build settings, or stop with a message naming every bad variable but none of its values."""
    try:
        return settings_class(_env_file=env_file)
    except ValidationError as error:
        problems = "; ".join(
            f"{'.'.join(str(part) for part in item['loc']).upper()}: {item['msg']}"
            for item in error.errors(include_input=False, include_context=False)
        )
    raise ConfigError(f"Invalid configuration for {settings_class.__name__}: {problems}")
