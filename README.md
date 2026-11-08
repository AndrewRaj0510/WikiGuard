# WikiGuard

A live Wikipedia vandalism detector, designed as a production-style MLOps system that runs entirely on free services.

> **Status: Phase 2 models trained and evaluated.** Five models are trained on a versioned dataset of about 95,000 edits, and the data pipeline runs on a laptop, with live edits going through Redpanda to a parquet archive. Phase 3's scorer and API are built, and the whole pipeline runs locally in Docker Compose against recorded Wikipedia data. Nothing is deployed, there's no UI yet, and Explain hasn't been measured against a real LLM. Model results are from a held-out test window, not live traffic. This README is updated as each phase finishes.

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

## Models

Five neural networks are compared on the same data. The champion is chosen by evaluation and is never hard-coded. **It predicts whether an edit is likely to be reverted, which is broader than vandalism.** In a sample of 200 reverts, about one in five looked like deliberate damage, and nearly half looked like good-faith or low-quality edits. The score decides what a person looks at first, and a person always decides.

Results on the test window, the newest week: 25,302 edits, 5.89% reverted within 48 hours. All scores come from the quantized file that would be served.

| Model | Role | PR-AUC (95% interval) | Precision in top 50 | Size | Time per edit, p50 / p95 on a laptop CPU thread |
|---|---|---|---|---|---|
| RoBERTa-base | **Champion** | 0.364 (0.339 to 0.387) | 0.96 | 125 MB | 21 / 159 ms |
| DistilBERT | First fallback | 0.323 (0.299 to 0.348) | 0.82 | 67 MB | 11 / 79 ms |
| BiLSTM + GloVe | Second fallback | 0.307 (0.284 to 0.331) | 0.70 | 3.6 MB | 0.3 / 1.3 ms |
| BERT-base | Third fallback | 0.347 (0.321 to 0.370) | 0.98 | 110 MB | 21 / 159 ms |
| MLP (TF-IDF + metadata) | Last-resort fallback | 0.269 (0.247 to 0.292) | 0.58 | 2.6 MB | 0.1 / 0.2 ms |
| Rule baseline | Number to beat | 0.187 (0.171 to 0.203) | 0.28 | | |
| Constant score | Number to beat | 0.059 | | | |

BERT scores second, but ranks behind the healthy models: quantizing it moved its score by more than the project's 0.005 limit, so tests on the original model stop predicting the served one. On the same 2,000 edits, RoBERTa's PR-AUC is 0.391 and that of Wikimedia's [RevertRisk](https://meta.wikimedia.org/wiki/Machine_learning_models/Production/Language-agnostic_revert_risk) 0.253. RevertRisk wasn't trained on this window or this exact label, so that's a benchmark, not a verdict. Under limits of 0.1 CPU and 512 MB, on this laptop's CPU, RoBERTa peaks at 283 MB and takes 3.4 s at p95.

## Stack

Everything runs on free tiers except OpenAI API credits for the Explain feature. When a limit is reached, the app degrades rather than billing. Limits are confirmed on signup, and some choices are still open.

| Purpose | Service |
|---|---|
| Frontend | Next.js on Vercel (Hobby) |
| API | FastAPI on Render (free) |
| Database | Postgres on Supabase (free) |
| Streaming | Redpanda (self-hosted) |
| Cluster | k3s on Oracle Cloud Always Free ARM |
| Archive and backups | Oracle Object Storage (Always Free) |
| Experiments and registry | MLflow and DVC on DagsHub (free) |
| GPU training | A laptop GPU for the first runs; Colab or Kaggle (free) for retraining |
| Monitoring | Prometheus, Grafana, and Evidently (self-hosted) |
| CI/CD | GitHub Actions and Argo CD |
| LLM | OpenAI API on prepaid credits, with Gemini free tier as backup |
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

![PR-AUC with 95% intervals for five models and two baselines on the test window. RoBERTa is highest at 0.364 and a constant score lowest at 0.059.](docs/images/readme/report_leaderboard_2026-11-07.png)

![Precision against recall for all five models. The models differ most at low recall, where RoBERTa and BERT keep the highest precision, and they converge beyond about 50% recall.](docs/images/readme/eval_pr_curves_all_2026-11-07.png)

App and Grafana screenshots will appear here once those exist.

## Roadmap

| Phase | Goal | Status |
|---|---|---|
| 0 | Setup | Done |
| 1 | Data foundation | Done locally; not yet on Supabase or CI |
| 2 | Train and evaluate five models | Trained, evaluated, and registered on DagsHub; ARM benchmarks still open |
| 3 | Serving, API, and Explain | Built and tested locally. Not deployed, and Explain not yet measured on a real LLM |
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

Tests that need Postgres or Redpanda skip themselves unless pointed at a throwaway copy. Use `127.0.0.1` on Windows, because `localhost` can hang on IPv6:

```powershell
$env:TEST_DATABASE_URL = "postgresql://wikiguard:YOUR_PASSWORD@127.0.0.1:5433/wikiguard"  # its tables get wiped
$env:TEST_KAFKA_BOOTSTRAP = "127.0.0.1:19092"
.venv\Scripts\python -m pytest
```

Build the dataset, which can stop and resume safely. `WIKIMEDIA_USER_AGENT` and `USER_HASH_KEY` must be set in `.env`:

```powershell
.venv\Scripts\python scripts\backfill.py --target 2000 --out data\backfill_pilot   # about 15 minutes
.venv\Scripts\python scripts\backfill.py                                           # about 10 hours
.venv\Scripts\python scripts\check_backfill.py data\backfill
.venv\Scripts\python scripts\measure_capacity.py data\backfill                    # needs TEST_DATABASE_URL
.venv\Scripts\python scripts\plot_backfill.py data\backfill docs\images --browser "C:\Program Files\Google\Chrome\Application\chrome.exe"
```

What the backfill measured: 100,566 edits over 28 days, 6.21% reverted within 48 hours, and 12.3% of dated reverts coming later than that, which is why training uses dated labels and not the raw tag. Logged-out editors make 14% of labeled edits and 55% of reverted ones. Keeping edits for 3 days uses a projected 185 MB, or 37% of Supabase's free limit, at about 86,600 edits a day.

### Dataset snapshots

A snapshot is the exact data the models train on: `edits.parquet` and a `manifest.json` that records its time windows, counts, exclusions, and content hash. Git tracks a small `.dvc` pointer, and [DVC](https://dvc.org) stores the data on DagsHub. Training tools have their own environment, because MLflow's pins conflict with pre-commit's.

| Split | Window | Edits | Reverted within 48 hours |
|---|---|---|---|
| Train | up to Oct 19, 2026 | 44,411 | 6.67% |
| Validation | Oct 19 to 26 | 25,318 | 6.30% |
| Test | Oct 26 to Nov 2 | 25,302 | 5.89% |

These are the rows the models were trained and measured on. A snapshot can shrink after that, because rows Wikipedia has since removed are taken out before each push. The published copy has 12 fewer, none of them reverted, and its manifest records each removal and the previous hash.

```powershell
py -3.11 -m venv .venv-train
.venv-train\Scripts\python -m pip install -r jobs\trainer\requirements.txt

.venv\Scripts\python scripts\build_snapshot.py data\backfill snapshots
.venv\Scripts\python scripts\recheck_snapshot.py snapshots\NAME      # needs WIKIMEDIA_USER_AGENT; run before every push
.venv-train\Scripts\dvc add snapshots\NAME
.venv-train\Scripts\dvc status                                       # should list the snapshot, not "nothing tracked"
```

**Snapshots are published on DagsHub.** They contain text from Wikipedia, including offensive vandalism, but no page titles or editor hashes. The repo is public, but when tested on 2026-11-07 a download with no DagsHub login was refused, while MLflow's runs and registry were open to anyone. To push the first time, create an access token at [DagsHub](https://dagshub.com/user/settings/tokens). These commands store it in `.dvc/config.local`, which git ignores, and the prompt keeps it out of your shell history:

```powershell
$token = Read-Host "DagsHub token"
.venv-train\Scripts\dvc remote modify origin --local access_key_id $token
.venv-train\Scripts\dvc remote modify origin --local secret_access_key $token
.venv-train\Scripts\dvc push
```

If the push can't reach DagsHub, copy the endpoint from the repo's Remote button. Run the `Read-Host` line on its own: if you paste all the lines together, PowerShell takes the next line as the token.

**If Wikipedia hides a revision**, the public copy has to change too. The recheck removes the row locally, and then the old version must be deleted from DagsHub:

```powershell
.venv\Scripts\python scripts\recheck_snapshot.py snapshots\NAME
.venv-train\Scripts\dvc add snapshots\NAME
.venv-train\Scripts\dvc push
.venv-train\Scripts\dvc gc --workspace --cloud    # deletes everything the current pointers don't use, including local cached versions
```

### Training and evaluation

The trainer needs PyTorch, so it runs from `.venv-train`, with `PYTHONPATH=jobs/trainer` and a snapshot pulled by DVC. The first full run took about 3.5 hours on an RTX 3050 6 GB, with most of it the five transformer trials per model.

```powershell
.venv-train\Scripts\python -m trainer.train snapshots\NAME --model mlp --search    # or bilstm, bert, roberta
.venv-train\Scripts\python -m trainer.train snapshots\NAME --model distilbert --search --search-device cuda --device cpu --cap 30000 --epochs 1 --recipe '{"trainable_blocks":2}'
.venv-train\Scripts\python -m trainer.aging snapshots\NAME --params '{"dropout":0.3,"hidden":128,"learning_rate":0.0003,"use_comment":true}'
.venv-train\Scripts\python -m trainer.finish snapshots\NAME --register              # charts, model cards, aliases
```

Runs log to a local MLflow file, so no token is needed. To send the finished models to DagsHub, once `MLFLOW_TRACKING_*` is set in `.env`:

```powershell
.venv-train\Scripts\python scripts\publish_runs.py --dry-run                        # verifies every artifact, sends nothing
.venv-train\Scripts\python scripts\publish_runs.py                                  # about 1.5 GB for all five
```

Stop Docker Desktop before measuring latency on Windows, because it made transformer scoring about 7 to 10 times slower here. To measure under Render's limits:

```powershell
.venv-train\Scripts\python -m trainer.benchmark data\runs\mlp\model --rows snapshots\NAME --save-sample data\runs\benchmarks\sample.json
docker build -f jobs\trainer\Dockerfile.benchmark -t wikiguard-benchmark .
docker run --rm --cpus=0.1 --memory=512m -v "${PWD}\data\runs:/runs" wikiguard-benchmark /runs/roberta/model --rows /runs/benchmarks/sample.json --label render-limits
```

The golden set, `snapshots/golden-set-v1/`, is a frozen set of 200 edits that later retrains are checked against. Its labels were made by Claude and reviewed by the owner. It's tracked and pushed like a snapshot.

### Supabase

There are two Supabase projects, `prod` and `staging`, and the app uses the same two variable names for both. Each place the app runs holds its own values, and nothing gets added inside Supabase. It only supplies the connection strings. WikiGuard never uses the anon or `service_role` keys.

| Where the variables go | Which project | `DATABASE_URL` | `DATABASE_MIGRATIONS_URL` |
|---|---|---|---|
| `.env` on the laptop | `staging` | Transaction pooler, port 6543 | Session pooler, port 5432 |
| Staging service on Render, and staging cluster Secrets | `staging` | Transaction pooler, port 6543 | Only in the migration job |
| Prod service on Render, and prod cluster Secrets | `prod` | Transaction pooler, port 6543 | Only in the migration job |

Keep prod values out of `.env`. To migrate prod from the laptop, set it for one shell, which overrides `.env`:

```powershell
$env:DATABASE_MIGRATIONS_URL = "PROD SESSION POOLER STRING"
.venv\Scripts\alembic -c jobs\migrate\alembic.ini upgrade head
Remove-Item Env:DATABASE_MIGRATIONS_URL
```

Do these for each project, staging first:

1. Set `DATABASE_MIGRATIONS_URL`, then run `.venv\Scripts\alembic -c jobs\migrate\alembic.ini upgrade head`. It prints which project it's about to change.
2. In the project's SQL Editor, create the logins below, each with its own strong password, different in each project:

   ```sql
   CREATE ROLE labeler_login LOGIN PASSWORD 'choose-a-strong-password' IN ROLE wikiguard_labeler;
   ```

3. Give each service its own `DATABASE_URL`, with its login as the username: `labeler_login.PROJECT-REF`.

| Login | Member of | Used by | Needed from |
|---|---|---|---|
| `labeler_login` | `wikiguard_labeler` | Labeler | Phase 1 |
| `maintenance_login` | `wikiguard_maintenance` | Purge and retention | Phase 1 |
| `scorer_login` | `wikiguard_scorer` | Scorer | Phase 3 |
| `api_login` | `wikiguard_api` | API | Phase 3 |
| `trainer_login` | `wikiguard_trainer` | Retrain-request poller, then trainer and monitor | Phase 3 |

### Running the whole pipeline locally

`.env` needs its secrets first. Make the password hash with `scripts/hash_password.py`, and wrap it in single quotes in `.env`, because it contains `$` signs. Model files come from `data/runs`.

```powershell
docker compose up -d --build
docker compose run --rm labeler
docker compose run --rm retrain-poller
```

The API is then at `http://127.0.0.1:8000`, and its schema at `/v1/openapi.json`. This uses the live Wikimedia stream.

To test everything end to end, with recorded data, throwaway secrets, and no `.env`:

```powershell
.venv\Scripts\python scripts\compose_e2e.py
.venv\Scripts\python scripts\compose_e2e.py --render-limits   # also at 0.1 CPU and 512 MB
```

For the integration tests, start `postgres` and `redpanda`, and point `TEST_DATABASE_URL` at a throwaway database, because they delete data.

To measure Explain on a real model, which spends credits: `.venv\Scripts\python scripts\eval_explain.py --provider openai`.

More setup steps will be added as the project grows. The accounts and tools the project needs are listed at the start of Part 2 in [PLAN.md](docs/PLAN.md).

## Data and attribution

WikiGuard uses public data from Wikimedia. Wikipedia content is available under [CC BY-SA](https://creativecommons.org/licenses/by-sa/4.0/). Dataset snapshots include that text, so the same license applies to them, and each manifest says so. A revision's authors are listed through `https://en.wikipedia.org/w/index.php?diff=REV_ID`. WikiGuard is an independent project and isn't affiliated with the Wikimedia Foundation. Test fixtures in [tests/fixtures/wikimedia/](tests/fixtures/wikimedia/README.md) are scrubbed recordings, and that folder's README covers what was changed and the attribution for kept text.
