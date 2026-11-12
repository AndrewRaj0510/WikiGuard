# Build from the repo root: docker build -f services/archiver/Dockerfile .
# Two stages: the first installs and builds, the second is what runs, with no build tools in it.
# The base image is pinned by digest, because a tag can be moved to different contents.
FROM python:3.11-slim@sha256:07d4824036af339fa1b6ef769cc99d9c10a141769538b52b3b597c84f7a7b535 AS build

ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /build

# Requirements come before code, so a code change doesn't reinstall everything. The lock file
# holds a hash for every package, and pip refuses anything that doesn't match it.
COPY services/archiver/requirements.txt requirements.txt
RUN python -m venv /venv && /venv/bin/pip install --require-hashes -r requirements.txt

# The shared package is built into a wheel with locked build tools, then installed with no
# dependency lookup: everything it needs is already in the lock above.
COPY requirements-build.txt .
RUN pip install --require-hashes -r requirements-build.txt
COPY packages/wikiguard_core packages/wikiguard_core
RUN pip wheel --no-deps --no-build-isolation --wheel-dir /wheels ./packages/wikiguard_core && /venv/bin/pip install --no-deps /wheels/*.whl

FROM python:3.11-slim@sha256:07d4824036af339fa1b6ef769cc99d9c10a141769538b52b3b597c84f7a7b535

ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PATH=/venv/bin:$PATH
WORKDIR /app

COPY --from=build /venv /venv
COPY services/archiver/archiver.py .

# Local parquet files. Owned by the runtime user, because a new volume takes this folder's owner,
# and this user couldn't write to it otherwise (F-103).
ENV ARCHIVE_PATH=/archive
RUN mkdir -p /archive && chown 10001 /archive

USER 10001
HEALTHCHECK --interval=30s --timeout=5s --retries=3 CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')"
CMD ["python", "archiver.py"]
