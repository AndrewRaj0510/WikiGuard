# WikiGuard

A live Wikipedia vandalism detector, designed as a production-style MLOps system that runs entirely on free services.

> **Status: setup (Phase 0).** Local tooling and development services work, but the pipeline and models don't exist yet, so there are no results to show. This README is updated as the project is built.

## What it will do

1. Read English Wikipedia edits from Wikimedia's public live stream.
2. Score each edit for vandalism risk with a deep learning model.
3. Show the riskiest edits first in a review queue for human patrollers.
4. Offer a one-line LLM explanation of what an edit did.
5. Use community reverts as free labels, two days later, to measure live accuracy.
6. Retrain weekly by default (daily or every two weeks are options), and sooner if drift appears, promoting a new model only if it passes a strict gate.

WikiGuard never reverts anything itself. A person always decides.

## Architecture (planned)

```
Wikimedia EventStreams
  -> ingestor   filters article edits, fetches diffs, publishes to Redpanda
  -> scorer     runs the model fallback chain, writes scores to Postgres
  -> archiver   keeps a rolling 30 days in a parquet archive
  -> labeler    hourly: collects revert tags as labels
  -> monitor    daily: live metrics and drift; triggers retraining
  -> trainer    builds a snapshot, trains, evaluates, and promotes if better

api       FastAPI: review queue, edit detail, leaderboard, health, /predict, /explain
frontend  Next.js: live feed, review queue, edit detail, leaderboard, model health
```

## Models (planned)

Five neural networks are compared on the same data. The champion is chosen by evaluation and is never hard-coded.

| Model | Role |
|---|---|
| MLP (TF-IDF + metadata) | Baseline and last-resort fallback |
| BiLSTM + GloVe | Classic sequence model; fallback |
| DistilBERT | Fast transformer |
| BERT-base | Reference transformer |
| RoBERTa-base | Expected to perform best (to be tested) |

Model results will appear in [docs/metrics.md](docs/metrics.md) once measured.

## Stack

Everything runs on free tiers, and the total cost must stay $0. When a free limit is reached, the app degrades rather than billing. Limits are confirmed on signup, and some choices are still open.

| Purpose | Service |
|---|---|
| Frontend | Next.js on Vercel (Hobby) |
| API | FastAPI on Render (free) |
| Database | Postgres on Supabase (free) |
| Streaming | Redpanda (self-hosted) |
| Cluster | k3s on Oracle Cloud Always Free ARM |
| Archive and backups | Oracle Object Storage (Always Free) |
| Experiments and registry | MLflow and DVC on DagsHub (free) |
| GPU training | Colab or Kaggle (free) |
| Monitoring | Prometheus, Grafana, and Evidently (self-hosted) |
| CI/CD | GitHub Actions and Argo CD |
| LLM | Groq free tier, with Gemini free tier as backup |
| Edge protection | Cloudflare Worker (free) |
| Admin access | Tailscale (free) |
| Errors and alerts | Sentry, Discord, UptimeRobot, and Healthchecks.io (free tiers) |

## Access (planned)

The demo allows one person at a time, in one tab, which keeps it within free limits. The landing page is always open, even when the app is in use.

| Account | Sign-in | Can do |
|---|---|---|
| `guest` | Public, one click | View the feed, queue, charts, and explanations. Can't approve edits or change anything. Sessions are limited to 15 minutes. |
| `user` | Password, owner only | Everything, including approving edits and retraining or rolling back models. |

## Results and screenshots

Nothing to show yet. Charts, matrices, and app and Grafana screenshots will appear here as real results exist, from `docs/images/readme/`.

## Roadmap

| Phase | Goal | Status |
|---|---|---|
| 0 | Setup | In progress |
| 1 | Data foundation | Not started |
| 2 | Train and evaluate five models | Not started |
| 3 | Serving, API, and Explain | Not started |
| 4 | Frontend | Not started |
| 5 | Monitoring and continuous training | Not started |
| 6 | CI/CD and security | Not started |
| 7 | Kubernetes and deployment | Not started |
| 8 | Production hardening | Not started |
| 9 | Polish and demo | Not started |

## Documentation

These documents are kept locally and aren't in this repo, so the links below only work in a local copy. Only the images the README uses are tracked, in `docs/images/readme/`.

| Document | What's in it |
|---|---|
| [PLAN.md](docs/PLAN.md) | Full design and task checklist |
| [walkthrough.md](docs/walkthrough.md) | Every step from first to last, and why |
| [decision.md](docs/decision.md) | Choices made and alternatives rejected |
| [findings.md](docs/findings.md) | What was discovered, with evidence |
| [evaluation.md](docs/evaluation.md) | How models are judged |
| [metrics.md](docs/metrics.md) | Measured results |
| [interview_guide.md](docs/interview_guide.md) | How to explain the project, with questions and answers |
| [memory.md](docs/memory.md) | Project rules and current state |

## Getting started

You need Python 3.11 and Docker Desktop. These commands are for Windows PowerShell.

```powershell
py -3.11 -m venv .venv
.venv\Scripts\python -m pip install -r requirements-dev.txt
.venv\Scripts\pre-commit install
.venv\Scripts\python -m pytest

# Start local Postgres (port 5433) and Redpanda (port 19092), bound to localhost only
Copy-Item .env.example .env   # then set POSTGRES_PASSWORD in .env
docker compose up -d --wait
```

Setup steps will be added here when there's something to run. The accounts and tools the project needs are listed at the start of Part 2 in [PLAN.md](docs/PLAN.md).

## Data and attribution

WikiGuard will use public data from Wikimedia. Wikipedia content is available under [CC BY-SA](https://creativecommons.org/licenses/by-sa/4.0/). WikiGuard is an independent project and isn't affiliated with the Wikimedia Foundation.
