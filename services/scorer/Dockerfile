# Build from the repo root: docker build -f services/scorer/Dockerfile .
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
WORKDIR /app

# Requirements come before code, so a code change doesn't reinstall everything.
COPY services/scorer/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY packages/wikiguard_core packages/wikiguard_core
RUN pip install --no-cache-dir --no-deps ./packages/wikiguard_core
COPY services/scorer/scorer.py .

# Verified models live here. A mounted volume in the cluster, so a restart skips the download.
ENV MODEL_CACHE_DIR=/models/cache
# ONNX Runtime writes a small file under HOME, and this user has no home folder.
ENV HOME=/tmp
RUN mkdir -p /models/cache && chown -R 10001 /models
USER 10001
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')"
CMD ["python", "scorer.py"]
