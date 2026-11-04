import re
from pathlib import Path

import pytest
from pydantic import SecretStr
from wikiguard_core import config
from wikiguard_core.config import (
    ArchiveSettings,
    BaseConfig,
    ConfigError,
    FeatureFlagSettings,
    LlmSettings,
    PrivacySettings,
    RetrainSettings,
    load_settings,
)

ENV_EXAMPLE = Path(__file__).parents[3] / ".env.example"
COMPOSE_ONLY = {"POSTGRES_USER", "POSTGRES_PASSWORD", "POSTGRES_DB", "POSTGRES_PORT"}


def settings_classes() -> list[type[BaseConfig]]:
    return [
        item
        for item in vars(config).values()
        if isinstance(item, type) and issubclass(item, BaseConfig) and item is not BaseConfig
    ]


def field_names() -> set[str]:
    return {name.upper() for cls in settings_classes() for name in cls.model_fields}


def example_values() -> dict[str, str]:
    matches = re.findall(r"^([A-Z0-9_]+)=(.*)$", ENV_EXAMPLE.read_text(), flags=re.MULTILINE)
    return dict(matches)


@pytest.fixture(autouse=True)
def clean_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep a developer's real environment out of these tests."""
    for name in field_names():
        monkeypatch.delenv(name, raising=False)


def test_missing_required_values_are_all_named() -> None:
    with pytest.raises(ConfigError) as error:
        load_settings(LlmSettings, env_file=None)
    for name in ("LLM_MODEL", "GROQ_API_KEY", "GEMINI_API_KEY", "EXPLAIN_DAILY_CALL_CAP"):
        assert name in str(error.value)


def test_blank_required_value_counts_as_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("USER_HASH_KEY", "")
    with pytest.raises(ConfigError, match="USER_HASH_KEY"):
        load_settings(PrivacySettings, env_file=None)


def test_error_never_includes_the_value(monkeypatch: pytest.MonkeyPatch) -> None:
    short_secret = "secret-value-that-is-too-short"
    monkeypatch.setenv("USER_HASH_KEY", short_secret)
    with pytest.raises(ConfigError) as error:
        load_settings(PrivacySettings, env_file=None)
    assert short_secret not in str(error.value)
    assert error.value.__cause__ is None
    assert "USER_HASH_KEY" in str(error.value)


def test_secret_is_hidden_when_printed(monkeypatch: pytest.MonkeyPatch) -> None:
    secret = "k" * 32
    monkeypatch.setenv("USER_HASH_KEY", secret)
    settings = load_settings(PrivacySettings, env_file=None)
    assert secret not in repr(settings)
    assert settings.user_hash_key.get_secret_value() == secret


@pytest.mark.parametrize("days", ["1", "7", "14"])
def test_supported_retrain_intervals(monkeypatch: pytest.MonkeyPatch, days: str) -> None:
    monkeypatch.setenv("RETRAIN_INTERVAL_DAYS", days)
    assert load_settings(RetrainSettings, env_file=None).retrain_interval_days == int(days)


@pytest.mark.parametrize("days", ["0", "2", "30", "weekly"])
def test_unsupported_retrain_intervals(monkeypatch: pytest.MonkeyPatch, days: str) -> None:
    monkeypatch.setenv("RETRAIN_INTERVAL_DAYS", days)
    with pytest.raises(ConfigError, match="RETRAIN_INTERVAL_DAYS"):
        load_settings(RetrainSettings, env_file=None)


def test_flag_can_be_switched_off(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GATE_ENABLED", "false")
    assert load_settings(FeatureFlagSettings, env_file=None).gate_enabled is False


def test_archive_window_cannot_delete_unlabeled_edits(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ARCHIVE_RETENTION_DAYS", "1")
    with pytest.raises(ConfigError, match="ARCHIVE_RETENTION_DAYS"):
        load_settings(ArchiveSettings, env_file=None)


def test_settings_read_from_env_file(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("RETRAIN_INTERVAL_DAYS=14\nUNRELATED_SETTING=ignored\n")
    assert load_settings(RetrainSettings, env_file=str(env_file)).retrain_interval_days == 14


def test_example_lists_every_setting() -> None:
    assert set(example_values()) == field_names() | COMPOSE_ONLY


def test_example_has_no_secret_values() -> None:
    secret_fields = {
        name.upper()
        for cls in settings_classes()
        for name, field in cls.model_fields.items()
        if SecretStr in (field.annotation, *getattr(field.annotation, "__args__", ()))
    }
    values = example_values()
    assert secret_fields
    assert {
        name: values[name] for name in secret_fields | {"POSTGRES_PASSWORD"} if values[name]
    } == {}
