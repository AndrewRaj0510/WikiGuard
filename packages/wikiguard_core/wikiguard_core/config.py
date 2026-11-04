"""Validated settings, read from the environment or a local .env file.

Settings are split by area so each service asks only for what it uses. The ingestor can then
start without the LLM key, and a leak in one service exposes fewer secrets. A service combines
the areas it needs and calls load_settings() once at startup:

    class ScorerSettings(CommonSettings, DatabaseSettings, KafkaSettings, ServingSettings):
        pass

    settings = load_settings(ScorerSettings)
"""

from typing import Literal, TypeVar

from pydantic import Field, SecretStr, ValidationError, field_validator
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


class ApiSettings(BaseConfig):
    session_secret_key: SecretStr = Field(min_length=MIN_SECRET_KEY_LENGTH)
    user_password_hash: SecretStr
    metrics_bearer_token: SecretStr
    cors_allowed_origins: str = "http://localhost:3000"


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

    @field_validator("retrain_interval_days")
    @classmethod
    def check_supported_interval(cls, value: int) -> int:
        if value not in {1, 7, 14}:
            raise ValueError("must be 1, 7, or 14")
        return value


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
