# Build from the repo root: docker build -f jobs/migrate/Dockerfile .
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
WORKDIR /app

COPY jobs/migrate/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY packages/wikiguard_core packages/wikiguard_core
RUN pip install --no-cache-dir --no-deps ./packages/wikiguard_core
COPY jobs/migrate jobs/migrate

USER 10001
CMD ["alembic", "-c", "jobs/migrate/alembic.ini", "upgrade", "head"]
