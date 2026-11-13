# The retraining Job (DEC-072): the CPU models train here, in the cluster, never on Render.
# Build from the repo root: docker build -f jobs/trainer/Dockerfile .
# Two stages: the first installs and builds, the second is what runs, with no build tools in it.
# The base image is pinned by digest, because a tag can be moved to different contents.
FROM python:3.11-slim@sha256:07d4824036af339fa1b6ef769cc99d9c10a141769538b52b3b597c84f7a7b535 AS build

ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /build

# BuildKit sets TARGETARCH to amd64 or arm64. PyTorch's CPU wheel differs between them, so each
# has its own lock, both hash-checked. Requirements come before code, so a code change doesn't
# reinstall PyTorch.
ARG TARGETARCH
COPY jobs/trainer/requirements-job-amd64.txt jobs/trainer/requirements-job-arm64.txt ./
RUN python -m venv /venv && /venv/bin/pip install --require-hashes -r "requirements-job-${TARGETARCH}.txt"

# The shared package and the trainer are built into wheels with locked build tools, then installed
# with no dependency lookup: everything they need is already in the lock above.
COPY requirements-build.txt .
RUN pip install --require-hashes -r requirements-build.txt
COPY packages/wikiguard_core packages/wikiguard_core
COPY jobs/trainer/pyproject.toml jobs/trainer/pyproject.toml
COPY jobs/trainer/trainer jobs/trainer/trainer
RUN pip wheel --no-deps --no-build-isolation --wheel-dir /wheels ./packages/wikiguard_core ./jobs/trainer && /venv/bin/pip install --no-deps /wheels/*.whl

FROM python:3.11-slim@sha256:07d4824036af339fa1b6ef769cc99d9c10a141769538b52b3b597c84f7a7b535

ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PATH=/venv/bin:$PATH

# git, because DVC works inside a checkout: the Job's first step clones the public repo and pulls
# the golden set, and retention removes snapshots no live model uses.
RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

COPY --from=build /venv /venv

# /work is the Job's cache volume: the repo checkout, GloVe, Hugging Face files, and the run's
# folder. All three downloads are large, so they're kept between runs. Nothing else is written.
WORKDIR /work
ENV HOME=/tmp \
    HF_HOME=/work/hf \
    MPLCONFIGDIR=/tmp/matplotlib \
    SNAPSHOT_REPO_PATH=/work/repo
RUN mkdir -p /work && chown 10001 /work
USER 10001
CMD ["python", "-m", "trainer.lifecycle", "--help"]
