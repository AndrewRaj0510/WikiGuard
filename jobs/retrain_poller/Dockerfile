# Build from the repo root: docker build -f jobs/retrain_poller/Dockerfile .
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
WORKDIR /app

COPY jobs/retrain_poller/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY packages/wikiguard_core packages/wikiguard_core
RUN pip install --no-cache-dir --no-deps ./packages/wikiguard_core
COPY jobs/retrain_poller/poller.py .

USER 10001
CMD ["python", "poller.py"]
