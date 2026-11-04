# Build from the repo root: docker build -f services/ingestor/Dockerfile .
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
WORKDIR /app

# Requirements come before code, so a code change doesn't reinstall everything.
COPY services/ingestor/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY packages/wikiguard_core packages/wikiguard_core
RUN pip install --no-cache-dir --no-deps ./packages/wikiguard_core
COPY services/ingestor/ingestor.py .

# Holds the stream position. A mounted volume in the cluster.
RUN mkdir -p data && chown 10001 data

USER 10001
HEALTHCHECK --interval=30s --timeout=5s --retries=3 CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')"
CMD ["python", "ingestor.py"]
