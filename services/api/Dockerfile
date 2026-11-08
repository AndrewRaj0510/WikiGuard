# Build from the repo root: docker build -f services/api/Dockerfile .
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
WORKDIR /app

# Requirements come before code, so a code change doesn't reinstall everything.
COPY services/api/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY packages/wikiguard_core packages/wikiguard_core
RUN pip install --no-cache-dir --no-deps ./packages/wikiguard_core
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
