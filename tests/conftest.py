"""Real-Postgres fixtures for repo tools: the same ones the jobs use."""

import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "jobs_conftest_for_tests", Path(__file__).parents[1] / "jobs" / "conftest.py"
)
assert spec is not None and spec.loader is not None  # noqa: S101 - the file is in this repo
jobs_conftest = importlib.util.module_from_spec(spec)
spec.loader.exec_module(jobs_conftest)

migrated_url = jobs_conftest.migrated_url
db = jobs_conftest.db
