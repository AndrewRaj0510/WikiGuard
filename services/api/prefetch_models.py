"""Download Render's two models into the image while it builds (D3, DEC-019).

Render's disk is wiped each time the free service sleeps, and waking already takes about a minute,
so downloading RoBERTa on every wake would be slow and would make DagsHub a dependency of every cold
start. Run in the Dockerfile when PREFETCH_MODELS is true: it resolves the chain the API will serve
(the champion and the MLP, DEC-050), downloads both into MODEL_CACHE_DIR, checks each file against
its recorded SHA-256, and saves the manifest the chain falls back to when the registry can't be
reached. At startup the API still reads the aliases (DEC-055), finds the files already cached and
verified, and loads them without a download.

    MODEL_CACHE_DIR=/models/cache MODEL_CHAIN_MODE=champion_and_last python prefetch_models.py
"""

import logging
import sys
from pathlib import Path

from wikiguard_core.chain import ChainMode, ModelCache, source_from_settings
from wikiguard_core.config import ConfigError, ServingSettings, load_settings

logger = logging.getLogger("prefetch")


def main() -> int:
    logging.basicConfig(level=logging.INFO)
    try:
        settings = load_settings(ServingSettings, env_file=None)
    except ConfigError as error:
        logger.error("%s", error)
        return 2
    source = source_from_settings(settings)
    refs = source.resolve()
    if ChainMode(settings.model_chain_mode) is ChainMode.CHAMPION_AND_LAST and len(refs) > 1:
        refs = [refs[0], refs[-1]]
    cache = ModelCache(Path(settings.model_cache_dir))
    for ref in refs:
        folder = cache.ensure(ref, source)
        logger.info("Cached %s (%s v%s) in %s", ref.alias, ref.model_name, ref.version, folder)
    cache.save_manifest(refs)
    return 0


if __name__ == "__main__":
    sys.exit(main())
