"""Start the API from the environment. Bad settings stop it here, with a message naming them.

uvicorn wikiguard_api.main:build --factory --workers 1
"""

import logging
import sys

from fastapi import FastAPI
from wikiguard_core.chain import chain_from_settings
from wikiguard_core.config import ConfigError, LlmSettings, load_settings
from wikiguard_core.errors import init_error_tracking
from wikiguard_core.identity import RedactingFilter

from wikiguard_api.app import create_app
from wikiguard_api.explain import GeminiProvider, OpenAiProvider, Provider
from wikiguard_api.foundations import JsonFormatter, Settings


def configure_logging(level: str) -> None:
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    handler.addFilter(RedactingFilter())
    logging.basicConfig(level=level, handlers=[handler], force=True)
    # httpx logs every URL at INFO, and nothing needs that.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.access").disabled = True


def build() -> FastAPI:
    try:
        settings = load_settings(Settings)
        configure_logging(settings.log_level)
        init_error_tracking(
            settings.sentry_dsn, "api", settings.environment, settings.git_commit or None
        )
        providers: list[Provider] = []
        cap = 0
        if settings.explain_enabled:
            llm = load_settings(LlmSettings)
            cap = llm.explain_daily_call_cap
            providers = [
                OpenAiProvider(llm.llm_model, llm.openai_api_key.get_secret_value()),
                GeminiProvider(llm.llm_backup_model, llm.gemini_api_key.get_secret_value()),
            ]
    except ConfigError as error:
        print(error, file=sys.stderr)  # noqa: T201 - logging may not be set up yet
        raise SystemExit(2) from None
    chain = chain_from_settings(settings) if settings.predict_enabled else None
    return create_app(settings, chain, providers, cap)
