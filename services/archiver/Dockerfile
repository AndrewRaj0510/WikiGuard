# Build from the repo root: docker build -f services/archiver/Dockerfile .
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
WORKDIR /app

# Requirements come before code, so a code change doesn't reinstall everything.
COPY services/archiver/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY packages/wikiguard_core packages/wikiguard_core
RUN pip install --no-cache-dir --no-deps ./packages/wikiguard_core
COPY services/archiver/archiver.py .

# Local parquet files. Owned by the runtime user, because a new volume takes this folder's owner,
# and this user couldn't write to it otherwise (F-103).
ENV ARCHIVE_PATH=/archive
RUN mkdir -p /archive && chown 10001 /archive

USER 10001
HEALTHCHECK --interval=30s --timeout=5s --retries=3 CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')"
CMD ["python", "archiver.py"]
