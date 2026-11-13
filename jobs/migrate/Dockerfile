# Build from the repo root: docker build -f jobs/migrate/Dockerfile .
# Two stages: the first installs and builds, the second is what runs, with no build tools in it.
# The base image is pinned by digest, because a tag can be moved to different contents.
FROM python:3.15-slim@sha256:65a65fc7b30b06cb6df4930fd4650697cb23f4c2698ab81fc87e4b031bc36928 AS build

ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /build

# Requirements come before code, so a code change doesn't reinstall everything. The lock file
# holds a hash for every package, and pip refuses anything that doesn't match it.
COPY jobs/migrate/requirements.txt requirements.txt
RUN python -m venv /venv && /venv/bin/pip install --require-hashes -r requirements.txt

# The shared package is built into a wheel with locked build tools, then installed with no
# dependency lookup: everything it needs is already in the lock above.
COPY requirements-build.txt .
RUN pip install --require-hashes -r requirements-build.txt
COPY packages/wikiguard_core packages/wikiguard_core
RUN pip wheel --no-deps --no-build-isolation --wheel-dir /wheels ./packages/wikiguard_core && /venv/bin/pip install --no-deps /wheels/*.whl

FROM python:3.15-slim@sha256:65a65fc7b30b06cb6df4930fd4650697cb23f4c2698ab81fc87e4b031bc36928

ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PATH=/venv/bin:$PATH
WORKDIR /app

COPY --from=build /venv /venv
COPY jobs/migrate jobs/migrate

USER 10001
CMD ["alembic", "-c", "jobs/migrate/alembic.ini", "upgrade", "head"]
