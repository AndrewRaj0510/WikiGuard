# Build from the repo root: docker build -f jobs/purge/Dockerfile .
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
WORKDIR /app

# Requirements come before code, so a code change doesn't reinstall everything.
COPY jobs/purge/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY packages/wikiguard_core packages/wikiguard_core
RUN pip install --no-cache-dir --no-deps ./packages/wikiguard_core
COPY jobs/purge/purge.py .

USER 10001
CMD ["python", "purge.py"]
