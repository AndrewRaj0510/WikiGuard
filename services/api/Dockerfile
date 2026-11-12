# Build from the repo root: docker build -f services/api/Dockerfile .
# Two stages: the first installs and builds, the second is what runs, with no build tools in it.
# The base image is pinned by digest, because a tag can be moved to different contents.
FROM python:3.11-slim@sha256:07d4824036af339fa1b6ef769cc99d9c10a141769538b52b3b597c84f7a7b535 AS build

ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /build

# Requirements come before code, so a code change doesn't reinstall everything. The lock file
# holds a hash for every package, and pip refuses anything that doesn't match it.
COPY services/api/requirements.txt requirements.txt
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
COPY services/api/wikiguard_api wikiguard_api

# Verified models live here. Render's disk doesn't last, so Phase 7 builds its models into the image.
ENV MODEL_CACHE_DIR=/models/cache
# ONNX Runtime writes a small file under HOME, and this user has no home folder.
ENV HOME=/tmp
RUN mkdir -p /models/cache && chown -R 10001 /models
USER 10001
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')"
# One worker: rate limits and caches are in process (DEC-016). SIGTERM finishes in-flight requests.
CMD ["uvicorn", "wikiguard_api.main:build", "--factory", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--timeout-graceful-shutdown", "20", "--no-server-header", "--no-access-log"]
