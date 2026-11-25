<div align="center">

# WikiGuard

WikiGuard is a real-time platform that watches every edit made to English Wikipedia articles and uses deep learning models to score how likely each one is to be vandalism. It ranks the most suspicious edits at the top of a review queue, so volunteer patrollers can check them first, with a one-click AI summary of what each edit changed. It learns from the edits Wikipedia's community undoes, tracks its own accuracy, and retrains itself when it starts to slip.

[![CI](https://github.com/AndrewRaj0510/WikiGuard/actions/workflows/ci.yml/badge.svg)](https://github.com/AndrewRaj0510/WikiGuard/actions/workflows/ci.yml)
![Python 3.11](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-models-EE4C2C?logo=pytorch&logoColor=white)
![ONNX Runtime](https://img.shields.io/badge/ONNX-int8%20serving-005CED?logo=onnx&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-API-009688?logo=fastapi&logoColor=white)
![Next.js 16](https://img.shields.io/badge/Next.js-16-000000?logo=nextdotjs&logoColor=white)
![Cost: $0](https://img.shields.io/badge/infrastructure-free%20tiers-2ea44f)

[Live site](https://wiki-guard.vercel.app) · [Results](#results-at-a-glance) · [Architecture](#architecture) · [Models](#models) · [Retraining](#the-retraining-story) · [Security](#security) · [Known limits](#known-limits) · [Development](#development)

</div>

> **About the live site.** [wiki-guard.vercel.app](https://wiki-guard.vercel.app) serves the landing and privacy pages. The console (queue, feed, models, health) needs the API on Render and the Cloudflare Worker, which aren't deployed yet, so for now it says it can't connect. Once they are, it serves a scored snapshot of real edits and says so on every page. It never shows live data ([why](#project-status)).

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="frontend/public/screenshots/queue-dark.webp">
  <img alt="The WikiGuard review queue: edits ranked by risk, each with a band, a score, a Logged out label where it applies, and the model that scored it." src="frontend/public/screenshots/queue-light.webp">
</picture>

## What WikiGuard does

Anyone can edit Wikipedia, and English Wikipedia articles get about 86,600 edits a day. Most of them are fine. Some are not: a paragraph deleted, a name swapped for an insult, a date quietly changed. Volunteers called patrollers catch these by reading the stream of recent changes, but nobody can read all of them, so some damage stays up until someone happens to notice it.

WikiGuard helps patrollers decide where to look first. It works in four steps:

1. **Watch.** It reads every edit to an English Wikipedia article as it happens, from Wikimedia's public live feed.
2. **Score.** A neural network reads what the edit removed and what it added, and gives it a risk score.
3. **Rank.** The riskiest edits go to the top of a review queue. A patroller opens one, sees exactly what changed, and can click **Explain** to get a one-line, plain-English summary of the edit from an LLM. WikiGuard never undoes anything itself. A person always decides.
4. **Learn.** When the community undoes an edit, Wikipedia tags it as reverted. Two days after each edit, WikiGuard checks for that tag, so it finds out whether its guess was right without anyone labelling data by hand. It uses those answers to check its own accuracy every day, and it retrains itself on a schedule or when new edits start to look different from the ones it learned from. A new model replaces the current one only if it is clearly better, and the old one comes back automatically if the new one does worse on live edits.

> **What the score means.** WikiGuard learns from reverts, and not every revert is vandalism. In a sample of 200 reverted edits, about one in five looked like deliberate damage, and nearly half looked like honest mistakes or low-quality edits. So a high score means "a person should look at this", not "this is vandalism".

### What the project shows

WikiGuard is a portfolio project, built the way a production machine learning system would be, from the live data feed to the deployed app:

- **A real streaming pipeline.** Edits stream in from Wikimedia through Redpanda, a Kafka-compatible message queue, and a rolling 30-day archive is kept as parquet files. It runs on Kubernetes on my machine against the live stream. No free server can run it around the clock, so the public demo serves a scored snapshot of real edits instead, and says so on every page (see [Project status](#project-status)).
- **Five deep learning models, compared fairly.** An MLP, a BiLSTM with GloVe word vectors, DistilBERT, BERT, and RoBERTa, all in PyTorch. Each one trains on older edits and is tested on the newest ones, so no model sees the future. The best one serves, and the others step in if it fails. Which model is best is decided by the evaluation, never written into the code.
- **Every experiment on record.** MLflow tracks each training run and holds the model registry, and DVC versions each dataset snapshot, both hosted on DagsHub.
- **A model that watches itself.** Daily drift and accuracy reports, a strict test before a new model is promoted, a trial period where it scores live edits without being shown, and automatic rollback.
- **A web app built for untrusted text.** Edits can contain anything, so the Next.js frontend shows them only as plain text, under a strict content security policy, and is tested against a set of XSS attacks and for accessibility (WCAG 2.1 AA). The FastAPI backend checks the user's role on every request.
- **Real operations.** CI/CD on GitHub Actions with hash-locked dependencies and scanned images, a Helm chart run on Kubernetes (kind) with Pod Security and default-deny networking, and Prometheus, Grafana, and Alertmanager for monitoring.
- **Free to run.** Everything runs on free tiers. The only paid part is a small prepaid balance of OpenAI credits for the Explain button.

## Results at a glance

Every number below was measured. Where it was measured matters, so it's in the last column: nothing behind the frontend is deployed, so the service numbers come from the real images on a laptop, at Render's free-plan limits (0.1 CPU, 512 MB).

| What | Result | Where measured |
|---|---|---|
| Champion model | RoBERTa-base, test PR-AUC **0.364** (95% interval 0.339 to 0.387), against 0.187 for a rule baseline. **48 of the 50** riskiest edits were really reverted | Held-out newest week, 25,302 edits |
| Edit to score | p95 **19.0 s** against a 60 s target | Live Wikimedia stream, Kubernetes (kind) on the laptop |
| Stream at 5× the real rate | 3,726 of 3,726 edits scored, p95 **0.7 s**, no backlog | Replay of real edits, Docker Compose |
| Read endpoints | p95 **17.5 ms** normal, **20.5 ms** at 10×, against 200 ms | k6 at Render's limits |
| `/predict` with RoBERTa | p95 2.4 to 3.9 s against 300 ms: **missed** (see [Known limits](#known-limits)) | k6 at Render's limits |
| Retraining loop | Drift found → retrained → gate passed → shadow → promoted: PR-AUC **+0.012** (+0.008 to +0.018). Poisoned labels **rejected** before training. A bad model **rolled back** by the daily check | Replays on real edits in a sandbox |
| Failure injection | **10 of 10** failure scenarios behave as designed, after 6 fixes the tests found | Docker Compose, end to end |
| Penetration test | **29 of 30** attacks held (OWASP API and LLM Top 10); 1 open finding, 2 couldn't run locally | The demo's images behind the real Worker code |
| Secret rotation | Every secret the API holds rotated in 115 s, **0** visitors refused | Same |
| Leaked password incident | Contained **44 s** after it was found, recovered at 146 s | Same |
| Database restore | Demo database rebuilt from a backup in **22 to 27 s**, every table identical | Fresh local Postgres |
| Leaks | **0** secrets, tokens, usernames, or diff text in 1,107 log lines or 251 browser responses | Whole pipeline, and Chromium |
| Cost | **$0** apart from OpenAI credits; the single seat keeps the worst case inside every free limit | Usage per visitor, measured |

## Architecture

The full design: color shows where each part was meant to run. **What's deployed is smaller:** the frontend on Vercel, the API on Render, Postgres on Supabase, and the Worker. Everything marked k3s is built and runs on a local Kubernetes cluster (kind), not on a server, because Oracle's free VM never had capacity and no other free host is big enough. [Project status](#project-status) has the details.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/images/readme/architecture_dark_2026-11-11.svg">
  <img alt="WikiGuard architecture. Wikimedia's EventStreams feed the ingestor, which fetches diffs from the MediaWiki API and publishes edits to Redpanda. The scorer writes scores to Postgres on Supabase, the archiver writes every topic to a parquet archive that's mirrored to Oracle Object Storage, and the hourly labeler turns revert tags into labels. Continuous training reads the archive and promotes models in the DagsHub registry, which the scorer and the FastAPI API on Render load. A patroller reaches one Cloudflare Worker address, which sends /api to Render and pages to the Next.js app on Vercel. Explain calls an LLM." src="docs/images/readme/architecture_light_2026-11-11.svg">
</picture>

| Component | Runs on | Role |
|---|---|---|
| `ingestor` | k3s | Reads the `recentchange` stream, keeps human edits to English Wikipedia articles, fetches each diff, and publishes it to `edits.raw` |
| `scorer` | k3s | Scores each edit with the champion, falling back down the chain when a model fails, and writes scores to Postgres and `edits.scored` |
| `archiver` | k3s | Writes edits, scores, and labels to compressed parquet on the cluster's disk for a rolling 30 days, mirrored nightly to Oracle Object Storage |
| `labeler` | k3s CronJob, hourly | Asks the MediaWiki API which edits got the `mw-reverted` tag, and finalizes each label at 48 hours |
| `monitor` | k3s CronJob, daily | Live metrics, drift, calibration, and slice metrics from the archive. Asks for a retrain or a rollback, and never trains itself |
| `retrain poller` and `trainer` | k3s CronJob, and a Job for each request | The poller picks up each request and starts the trainer as a Kubernetes Job, which builds a snapshot, trains candidates, runs the promotion gate and a shadow check, and updates the registry aliases |
| `backup` | k3s CronJob, nightly | Exports the tables that can't be rebuilt, encrypted to a public key the cluster can't decrypt with, to the bucket |
| `api` | Render | FastAPI: review queue, edit detail, reviews, model leaderboard, health, `/predict`, and `/explain`. Owns sessions and roles |
| `frontend` | Vercel | Next.js: landing page, live feed, review queue, edit detail with Explain, model pages, and health |
| `edge` | Cloudflare Worker | The single site address: `/api` goes to Render and everything else to Vercel, so session cookies stay first-party |

Every change to what serves goes one way: the monitor, or the owner's button in the app, adds a row to `retrain_requests`, and the poller in the cluster acts on it. So the public API never holds cluster credentials. Grafana stays private, and the app draws its own charts from the API.

### The continuous-training loop

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/images/readme/training_loop_dark_2026-11-11.svg">
  <img alt="The continuous-training loop: an edit is scored and queued, the community reverts or keeps it, and its label is final at 48 hours. A daily monitor computes live metrics and drift. Drift, decay, or the schedule start a retrain. The promotion gate uses a paired bootstrap, the golden set, and slices: if it fails the champion stays, and if it passes the candidate scores live edits in shadow, then is promoted. A post-promotion check rolls back automatically if it is worse live." src="docs/images/readme/training_loop_light_2026-11-11.svg">
</picture>

Retraining runs every `RETRAIN_INTERVAL_DAYS` days (7 by default, 1 or 14 as options), and sooner on drift or falling live accuracy. Only the trainer moves registry aliases; the monitor and the owner's button just add a request.

## The retraining story

A model that learns from reverts has to keep learning, because vandalism changes. The loop above was run end to end three times, each as a replay on the real backfilled edits, with the real code, in a sandbox registry and database. Nothing was deployed, so no retrain has run on live traffic. The sandbox chain was the BiLSTM as champion and the MLP as fallback, the two models that retrain on a CPU in a reasonable time. The transformers retrain from a GPU notebook.

**1. Drift leads to a better model.** A synthetic spam campaign was planted in 6,517 edits over the last 20 days: a link to a made-up domain, from logged-out editors, labeled reverted. The daily monitor found text drift and drift in `uppercase_ratio` (1 of 9 comparable features over its threshold), and asked for a retrain with the reason "text drift found".

![Each feature's drift against its own threshold, on a log scale. uppercase_ratio is past the threshold; the other eight features are below it.](docs/images/readme/monitor_drift_2026-11-04.png)

The poller started the trainer, which built a fresh snapshot (94,819 rows), checked it, retrained both models, and ran the promotion gate. The gate compares the candidate with the champion on the same newest window, with a paired bootstrap, and also checks the golden set (200 frozen edits), each slice, latency, and size. Every check passed. The candidate then scored the newest day in shadow (3,613 edits, 0 errors) and was promoted.

![Before vs after on the newest test window: overall PR-AUC 0.866 before and 0.879 after; logged-out editors 0.953 to 0.965; registered editors 0.087 to 0.082; small edits 0.337 to 0.364; large edits 0.969 to 0.970.](docs/images/readme/report_before_after_sim_drift_2026-11-04.png)

| Measure | Before (BiLSTM v1) | After (BiLSTM v4) |
|---|---|---|
| PR-AUC, newest window | 0.866 (0.856 to 0.877) | 0.879 (0.870 to 0.887) |
| Paired difference | | **+0.0123** (+0.0075 to +0.0175) |
| False positive rate, logged-out editors | 0.379 | 0.114 |
| False positive rate, registered editors | 0.0072 | 0.0005 |
| Golden set PR-AUC | 0.7949 | 0.7912, inside the 0.02 allowed drop |
| p95 latency, size | | 15.2 ms, 4.3 MB |

These numbers show the loop working, not model quality: the planted campaign makes the task easy, so they can't be compared with the leaderboard. Registered editors' PR-AUC fell by 0.0046, which is inside the gate's allowance for a slice.

**2. Poisoned labels are refused.** 2,035 edits on one day were relabeled as reverted, a coordinated burst of fake reverts. The trainer's data checks refused the snapshot (that day's revert rate was 63.5%, over twice the mean) 72 s after the request. Nothing trained, no alias moved, and an alert went out. The limit: this check sees a burst, not poisoning spread thinly over many days.

**3. A bad model is rolled back.** A BiLSTM trained on shuffled labels was forced into the champion slot. The next daily check measured its live PR-AUC at 0.059 on 25,319 labels, against 0.320 for the model it replaced, and asked for a rollback. The aliases were back in under 0.05 s, and the scorer swapped its chain back in 0.1 s. On a cluster, the worst case is 195 s plus downloading the champion, because the scorer checks the aliases every 60 s; that wasn't timed on a running cluster.

## Models

Five neural networks are compared on the same data. The champion is chosen by evaluation and is never hard-coded: the registry aliases `champion`, `fallback-1`, `fallback-2`, and so on define the serving chain.

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

PR-AUC is the main measure, because only about 6% of edits are reverted. BERT scores second but ranks behind the healthy models: quantizing it moved its score by more than the project's 0.005 limit, so tests on the original model stop predicting the served one. On the same 2,000 edits, RoBERTa's PR-AUC is 0.391 and that of Wikimedia's [RevertRisk](https://meta.wikimedia.org/wiki/Machine_learning_models/Production/Language-agnostic_revert_risk) 0.253. RevertRisk wasn't trained on this window or this exact label, so that's a benchmark, not a verdict. Under limits of 0.1 CPU and 512 MB, on this laptop's CPU, RoBERTa peaks at 283 MB and takes 3.4 s at p95.

![PR-AUC with 95% intervals for five models and two baselines on the test window. RoBERTa is highest at 0.364 and a constant score lowest at 0.059.](docs/images/readme/report_leaderboard_2026-11-07.png)

![Precision against recall for all five models. The models differ most at low recall, where RoBERTa and BERT keep the highest precision, and they converge beyond about 50% recall.](docs/images/readme/eval_pr_curves_all_2026-11-07.png)

**Calibration.** The score is shown to patrollers as a chance of being reverted, so it should mean that. Temperature scaling, fitted on the validation week, brought RoBERTa's expected calibration error on the test week from 0.027 to 0.016.

![RoBERTa's reliability diagram on the test week: share reverted against mean score in each bin, before and after temperature scaling (T = 1.29), next to the line of perfect calibration. ECE 0.027 before, 0.016 after.](docs/images/readme/eval_reliability_roberta_2026-11-07.png)

**Disguised text.** Vandals write around filters with leetspeak, look-alike letters, zero-width characters, spacing, and case. Text is normalized before every model sees it, which removes zero-width characters entirely. The other tricks still cost each model some PR-AUC on 500 reverted and 500 good edits; RoBERTa loses the most, up to 0.045 with spaced-out letters.

![PR-AUC lost by each of the five models to five kinds of disguised text. Zero-width characters cost nothing. RoBERTa loses the most on every other kind, up to 0.045 for spacing.](docs/images/readme/eval_robustness_all_2026-11-07.png)

## Data

- **Source:** Wikimedia's `recentchange` stream for live edits, and the MediaWiki Action API for diffs, tags, and a 30-day backfill. Scope: English Wikipedia, article namespace, edits only, no bots.
- **Label:** `1` if the revision is tagged `mw-reverted` within 48 hours, `0` if it is still unreverted after 48 hours.
- **Splits are by time,** never random: train on older edits, test on the newest week.
- **Privacy:** editor names and IP addresses are never shown. Editors are stored as a keyed hash, with a flag for logged-out and temporary accounts.

The backfill collected 100,566 edits over 28 days. 6.21% were reverted within 48 hours, and 12.3% of dated reverts came later than that, which is why training uses dated labels and not the raw tag. Logged-out editors made 14% of labeled edits and 55% of reverted ones.

| Split | Window | Edits | Reverted within 48 hours |
|---|---|---|---|
| Train | up to Oct 19, 2026 | 44,411 | 6.67% |
| Validation | Oct 19 to 26 | 25,318 | 6.30% |
| Test | Oct 26 to Nov 2 | 25,302 | 5.89% |

![The share of edits reverted within 48 hours, for each day from October 6 to November 2, 2026, between 4.7% and 7.6%, with the train, validation, and test windows shaded in time order.](docs/images/readme/snapshot_splits_by_day_2026-11-06.png)

These are the rows the models were trained and measured on. A snapshot can shrink after that, because rows Wikipedia has since removed are taken out before each push. The published copy has 12 fewer, none of them reverted, and its manifest records each removal and the previous hash.

## The app

![One edit's page as the owner: the text it removed and added side by side, the review buttons with their keyboard keys, the score on a meter, and the Explain panel.](docs/images/readme/app_edit_owner_light_2026-11-09.png)

![The Models page: a leaderboard of the five models with 95% intervals as whiskers, and their precision-recall curves.](docs/images/readme/app_models_light_2026-11-09.png)

The screenshots use the real models and their measured results, with recorded Wikipedia edits whose text was scrubbed for the test fixtures.

The demo allows one person at a time, in one tab, which keeps it within free limits. The landing page is always open, even when the app is in use. Roles are checked on the server for every request.

| Account | Sign-in | Can do |
|---|---|---|
| `guest` | Public, one click | View the feed, queue, charts, and explanations. Can't review edits or change anything. Sessions last up to 15 minutes. |
| `user` | Password, owner only | Everything, including reviewing edits and retraining, promoting, or rolling back models. |

Keyboard flow: `j`/`k` to move, Enter to open, `v`/`g`/`o`/`s` to review, `n`/`p` for the next or previous edit, `e` for Explain, and `?` for help.

## Operations and reliability

The cluster half ran for hours on Kubernetes (kind) on the laptop against the live Wikimedia stream, with Prometheus, Grafana, and Alertmanager. Grafana stays private, so the app draws its own charts from the API, and the dashboards are shared as screenshots.

![The Grafana Models dashboard from the kind run, three hours: edits scored per minute by model (RoBERTa about 62, DistilBERT about 9), the fallback rate near 10% against a 5% line, batch latency p95 of 280 ms for RoBERTa and 575 ms for DistilBERT, no circuit breakers open, and no shadow candidate loaded.](docs/images/readme/grafana_models_2026-11-12.png)

On kind, 74.5 edits a minute arrived from the live stream and were scored with a p95 of 19.0 s from the edit's time on Wikipedia. About 10% were answered by DistilBERT, because RoBERTa missed the scorer's 300 ms limit for one edit on the laptop. That is what the fallback chain is for.

**Failure injection.** Each row of the failure table was broken on purpose on the full Docker Compose stack (every real image, the real five models, and a mock Wikimedia with switches), with a browser watching the app. All 10 behave as designed, after six fixes the runs found; one of them is that a broken champion used to be invisible, because its fallback answers weren't flagged.

| Failure | What happened |
|---|---|
| Wikimedia stream down, 11.5 min | 29 reconnect tries with jittered backoff; "Stale" banner after 9.6 minutes; resumed from the last event, the gap scored 39 s later, no duplicates |
| Rate limits (429, `maxlag`) | Every `Retry-After` honored to within 0.05 s |
| Redpanda down | Services stayed up; all edits scored 48 s after it returned, no duplicates |
| Champion's file damaged | API ready in 6.3 s without it; answers come from the fallback and say so |
| Every model fails | `/predict` answers 503; the scorer stops, so edits wait in Redpanda instead of being stored unscored |
| Postgres down | Reads served stale with a banner until the session's 45 s lease ran out; scorer caught up 9.5 s after it returned |
| Model registry down | API and scorer start from the saved chain in 4.2 s |
| LLM down or over its daily cap | "Explanation unavailable"; the rest of the app unaffected |
| API asleep or down | Waking screen within 3.3 s; back without a reload |
| Cluster down | API and frontend keep serving stored data with a freshness banner |

![The screen the app shows while the API isn't answering: "WikiGuard isn't answering. It may be waking up, which can take a minute on the free plan. Still trying every few seconds." with a progress bar and a Try now button.](docs/images/readme/failure_api_down_1_2026-11-21.png)

## Service levels and load tests

Load tests ran with k6 against the API image at Render's free-plan limits (0.1 CPU, 512 MB) over the demo's 25,294 edits, with local Postgres, so Supabase's network time isn't included. The app allows one person at a time, so "load" means that person at the app's own limits, plus everyone else being turned away.

| Objective | Target | Measured | Met? |
|---|---|---|---|
| Edit to score | p95 ≤ 60 s | 19.0 s on the live stream (kind); 0.7 s replaying at 5× the real rate | Yes |
| Read endpoints | p95 ≤ 200 ms | 17.5 ms normal, 20.5 ms at 10×, paced like the frontend. Worst case, every poll on one clock: 295 and 188 ms | Yes, for real traffic |
| `/predict`, awake | p95 ≤ 300 ms | 2.4 to 3.9 s with RoBERTa answering (median about 380 ms). 183 to 435 ms with a 300 ms limit, but then the MLP answers 92% | No |
| Cold start | ≤ 90 s | 41.3 s to ready | Yes, in Docker; not on Render |
| Model rollback | < 5 min | 0.05 s to restore the aliases, 0.1 s to swap the chain (sandbox); 195 s plus a download worst case on a cluster | Yes in the sandbox; not timed on a cluster |
| Database restore | ≤ 4 h | 21.7 and 27.3 s for the demo's database | Yes, locally; not on Supabase |
| Ingestion uptime | ≥ 99% over 30 days | not measured: nothing runs around the clock | |

| Load test | Result |
|---|---|
| Normal reads, 5 minutes | p95 17.5 ms, p99 784 ms (the first polls of a page load arrive together) |
| 10× spike, 5 minutes | p95 20.5 ms, p99 95 ms; every extra visitor turned away by the seat, cheaply |
| One-hour soak | 2,657 requests, 0 failed checks, memory 263 → 240 MiB, latency flat |
| Explain burst, 30 edits at once | At most 4 LLM calls in flight, the rest answered "busy" at once; 20 requests for one edit made 1 LLM call |
| `/predict` at twice the owner's limit | Refused as designed: 429 from the rate limiter, 503 from the 2 slots |
| Stream replay at 5× (6.2 edits a second, 10 minutes) | All 3,726 scored; p50 0.2 s, p95 0.7 s, max 1.3 s; 1.7% by the fallback |

What the tests changed, measured before and after:

| Change | Before | After |
|---|---|---|
| `gc.freeze()` after startup, so a full garbage collection no longer walks the loaded models' objects | Reads p95 554 ms, p99 974 ms | 295 ms, 302 ms |
| At most 4 Explain LLM calls at once | 23 calls in flight in a 30-edit burst; feed p95 1,788 ms | 4 in flight; feed p95 1,472 ms |
| Lists cached 5 minutes in snapshot mode, cleared by any review | One seat on the feed for a month: about 10.8 GB from Supabase, against 5 GB free | About 1.2 GB |

## Security

Edits are written by anyone, including attackers, so every diff, title, comment, and username is treated as hostile, and so is the LLM's answer about them.

- **Text renders as text.** React text nodes only, under a nonce-based content security policy with no `unsafe-inline`. An LLM answer with markup or a link is refused, not shown.
- **Prompt injection.** The diff goes to the LLM as escaped data inside tags it can't close; the model has no tools, and its answer is checked before it's stored.
- **Access.** Two fixed accounts, `guest` and `user`, checked on the server for every request; one session in one tab across the whole app; CSRF tokens; Argon2 password hashing with lockouts per address and overall; an `HttpOnly` session cookie and nothing in browser storage.
- **Edge.** The API refuses anything that didn't come through the Cloudflare Worker, which also replaces any client-supplied address headers.
- **Data.** Supabase's Data API off, row-level security on every table, a least-privilege login per service, no editor IPs stored, editors kept as a keyed hash.
- **Supply chain.** Hash-locked Python installs, GitHub Actions pinned to commits, base images pinned by digest, CodeQL, Trivy, `pip-audit`, `npm audit`, gitleaks, and Dependabot. No pickled model files; checksums verified before loading.
- **Cluster.** Pod Security `restricted` on every pod, default-deny NetworkPolicies in both directions (13 of 13 connections allowed or refused as expected), one Secret per component, Sealed Secrets, nothing exposed. kubescape: 26 of 30 controls passed, the other 4 explained.

**Penetration test.** A script attacked the demo as it will be deployed: the API image at Render's limits on the demo's data, the built frontend, and the real Worker's code in front, with a fake LLM that could be told to answer with an attack. 32 attacks against the OWASP API Security Top 10 and the LLM Top 10:

| Area | Result |
|---|---|
| Calling the API around the Worker, spoofed address headers | Held: 403, and the Worker replaced every spoofed header |
| 14 console routes with no session; forged and reused cookies; missing CSRF tokens | Held: 401 and 403 |
| Password brute force, one address and many | Held: locked after 5 tries; 12 at once ran one hash check |
| A guest calling the 6 owner-only routes | Held: 403 |
| Oversized, malformed, and hostile input (11 kinds) | Held: never a 500, never a stack trace or SQL in a response. One fix: a streamed oversized body got 400 instead of 413 |
| Stored XSS: 6 payloads in a title, summary, and diff, in Chromium | Held: nothing ran, no CSP violation, every payload shown as text |
| Prompt injection and a hostile LLM answer | Held: escaped, and the answer refused |
| Using up the LLM budget | Held: guests get 5, the owner 10 a minute, and the daily cap stopped it at exactly 40 calls |
| A bot with two addresses taking turns in the guest seat | **Failed:** visitors were kept out (the owner can still take over). Open, see [Known limits](#known-limits) |
| Reading Supabase through its REST API; server-side request forgery | Not run: needs the real project; no route takes a URL |

![An edit page during the penetration test, through the Worker: the title, summary, and diff hold script tags, an img onerror, an svg onload, an iframe, and javascript: links, and every one of them shows as plain text.](docs/images/readme/pentest_xss_edit_2026-11-23.png)

The page above shows one of the two edits the test added to the demo's data, which is why the banner counts 25,296.

**Rehearsed.** A scheduled rotation of every secret the API holds took 115 s and refused no visitor, thanks to a short overlap for the Worker's secret. A leaked owner password was contained 44 s after it was found (read-only mode, everyone signed out, new password) and fully recovered at 146 s. Logs from the whole pipeline (1,107 lines) and every browser response held no secret, token, username, or diff text, including a frontend built with seven secrets in its environment.

## Known limits

Said plainly, so this README doesn't claim more than it should.

- **The demo isn't live, and isn't deployed yet.** No free host can run the streaming pipeline around the clock, so the public demo serves a fixed snapshot of real edits with their real labels, and says so on every page. The pipeline runs on Kubernetes on the laptop only. The API on Render, the Worker, and the demo's database on Supabase are built and tested locally and wait on my accounts.
- **`/predict` misses its 300 ms target** with RoBERTa on Render's tenth of a CPU (p95 2.4 to 3.9 s). A 300 ms limit meets it, but then the much weaker MLP answers most requests. Which one the demo serves is still to decide.
- **A bot with two IP addresses can hold the guest seat**, keeping visitors out. The owner can always take over. The options (a challenge, a waiting line, or accepting it) are written down; none is built.
- **No live model health yet.** Live PR-AUC, drift, and label lag are computed by the monitor, but have never run on live traffic. The retraining loop ran as replays, with the BiLSTM and MLP; the transformers retrain by hand from a GPU notebook.
- **The poisoning check sees bursts,** not poisoning spread thinly over many days.
- **BERT serves behind DistilBERT and the BiLSTM**, though it scores second, because its quantized file doesn't match the original closely enough to trust the tests.
- **An outage of the model registry in the middle of a retrain loses the run,** and sustained rate limiting from Wikimedia sends edits to the dead-letter queue instead of waiting.
- **Nothing outside watches or backs up the demo** yet: no uptime check on the API, and the backup job is a cluster CronJob, which isn't deployed.
- **Explain hasn't been scored on a real LLM.** The faithfulness and prompt-injection test sets exist; running them spends credits and is still to do.
- **Never run on ARM or on the Oracle VM.** The images are built for arm64 and the VM scripts are written, but Oracle never had free capacity.
- **Labels are reverts, not vandalism.** A high score means "a person should look at this".

## Tech stack

Everything runs on free tiers except OpenAI API credits for Explain. When a limit is reached, the app degrades rather than billing.

| Layer | Technology |
|---|---|
| Models | PyTorch (MLP, BiLSTM + GloVe, DistilBERT, BERT, RoBERTa), exported to ONNX with int8 quantization |
| Training | A laptop GPU for the first runs. Scheduled retraining runs the three CPU models on the cluster; BERT and RoBERTa retrain from a Colab or Kaggle notebook (free GPUs) |
| Experiments and data | MLflow registry and DVC on DagsHub |
| Monitoring | Evidently, Prometheus, Grafana, and Alertmanager |
| Streaming | Redpanda (Kafka API), self-hosted |
| Database | Postgres on Supabase, with row-level security on every table |
| Archive | Parquet on the cluster's disk, with a nightly mirror to S3-compatible storage (built for Oracle Object Storage; on the laptop it stays on a volume) |
| API | FastAPI on Render |
| Frontend | Next.js 16, React 19, Tailwind CSS 4, on Vercel |
| Edge | Cloudflare Worker |
| Cluster | Kubernetes on kind, locally. Designed for k3s on Oracle Cloud Always Free with Argo CD; those scripts are written but never ran, because Oracle had no capacity |
| CI/CD | GitHub Actions, GHCR, CodeQL, Trivy, pip-audit, gitleaks, Schemathesis, OWASP ZAP, Dependabot |
| LLM | OpenAI API on prepaid credits, with Gemini's free tier as backup |
| Alerts and admin | Sentry, Discord, UptimeRobot, Healthchecks.io, and Tailscale |

## Project status

The project is built phase by phase. Model results are from a held-out test window, not live traffic.

| Phase | Goal | Status |
|---|---|---|
| 0 | Setup | Done |
| 1 | Data foundation | Done locally; not yet on Supabase or CI |
| 2 | Train and evaluate five models | Trained, evaluated, and registered on DagsHub; ARM benchmarks still open |
| 3 | Serving, API, and Explain | Built and tested locally. Not deployed, and Explain not yet measured on a real LLM |
| 4 | Frontend | Done. Live on Vercel at [wiki-guard.vercel.app](https://wiki-guard.vercel.app); the console connects once the API is on Render (Phase 7) |
| 5 | Monitoring and continuous training | Built and tested locally. Drift-to-promotion and label poisoning replayed on real data in a sandbox; the rollback replay ran in Phase 8; anything live is still to run |
| 6 | CI/CD and security | Done: full CI on GitHub, hash-locked installs, multi-arch image pipeline, CodeQL, threat model. The deploy half of CD and staging stay switched off until the API is on Render |
| 7 | Kubernetes and deployment | Built and run on a local cluster (kind) against the live Wikimedia stream: Helm chart, Pod Security, default-deny networking, Prometheus alerts, Grafana dashboards, encrypted backups with a timed restore. The VM was dropped, so the cluster isn't deployed. The demo's data and its "snapshot, not live" label are built; Render and the Cloudflare Worker wait on my accounts |
| 8 | Production hardening | Done, locally (nothing behind the frontend is deployed): failure injection, load tests, a self-penetration test, secret rotation, incident, rollback, and restore rehearsals, a log and browser leak check, a runbook for every alert, and cost per visitor. Results are in [Operations](#operations-and-reliability), [Service levels](#service-levels-and-load-tests), and [Security](#security). The 7-day unattended run was dropped with the VM |
| 9 | Polish and demo | README done: results, retraining story, load tests, security, and known limits. The 2-minute demo video is next |

## Repository layout

```
frontend/                 Next.js app (Vercel)
services/api/             FastAPI public API (Render)
services/ingestor/        stream consumer
services/scorer/          scores edits from Kafka
services/archiver/        writes the parquet archive
jobs/labeler/             revert-label collector
jobs/monitor/             live metrics, drift, retrain and rollback requests
jobs/retrain_poller/      acts on retrain requests in the cluster
jobs/trainer/             snapshot, training, evaluation, promotion
jobs/migrate/             Alembic migrations: tables, roles, and row-level security
jobs/purge/               removes content Wikipedia has hidden
jobs/retention/           keeps Supabase under its free limit
packages/wikiguard_core/  shared code: schemas, features, diff parsing, the 5 models,
                          inference with the fallback chain, MLflow helpers
notebooks/                Colab and Kaggle notebooks for GPU training
jobs/backup/              nightly encrypted backup, and the restore
k8s/                      the Helm chart, platform values, Argo CD apps, and sealed secrets
edge/                     the Cloudflare Worker: the single site address
scripts/                  backfill, snapshots, evaluation, and one-off tools
snapshots/                DVC pointers to the dataset snapshots and the golden set
tests/                    tests and recorded Wikimedia fixtures
```

## Development

These commands are for Windows PowerShell. You need Python 3.11 and Docker Desktop, and Node.js 22 for the frontend.

### Setup and tests

```powershell
py -3.11 -m venv .venv
.venv\Scripts\python -m pip install -r requirements-build.txt -r requirements-dev.txt
.venv\Scripts\python -m pip install --no-deps --no-build-isolation -e packages\wikiguard_core
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

Tests never call Wikimedia or the LLM: they use recorded fixtures in `tests/fixtures/`.

### Dependencies

Every Python dependency is locked with its SHA-256. Each service and job has a `requirements.in`, the pins chosen by hand, and a generated `requirements.txt` that lists every package installed with its hashes. Images and CI install with `pip install --require-hashes`, which refuses a package that differs from the one that was locked. To change a dependency, edit the `.in` file and regenerate:

```powershell
.venv\Scripts\python scripts\lock_requirements.py              # every lock, keeping current versions
.venv\Scripts\python scripts\lock_requirements.py --upgrade    # also move indirect packages forward
.venv\Scripts\python scripts\lock_requirements.py --check      # what CI and pre-commit run; no network
```

The shared package is this repo's own code, so it can't be hash-locked. It's installed with `--no-deps`, after the lock that already holds its dependencies. `jobs/trainer/requirements.txt`, the laptop's GPU set, is the one list installed without hashes.

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

### Frontend

The app calls the API at `/api` on its own address, the way the Cloudflare Worker will serve it, so locally a Next.js rewrite sends `/api` to the API. Open it at `http://localhost:3000`, because the API's CORS allowlist names that address.

The lock file is written with npm 11.21.2, the version `package.json` names and CI and Dependabot use. Node 22's own npm 10 rejects it, so install that npm once: `npm install --global npm@11.21.2`.

```powershell
cd frontend
npm ci
npm test               # unit tests (Vitest)
npm run lint
npm run typecheck
$env:API_ORIGIN = "http://127.0.0.1:8000"   # read when building, never sent to the browser
npm run build
npm start

# Checks on a build. None needs a backend: the E2E tests mock the API in the browser.
npm run check:bundle                 # no secrets in client files; each page within its JS budget
npx playwright install chromium      # once
npm run test:e2e                     # flows, XSS corpus, axe, keyboard, headers
npx lhci autorun                     # Lighthouse budgets on the public pages
```

Every page renders per request so it can carry a fresh CSP nonce. To screenshot every page against the fixture stack, as guest and owner: `node scripts\capture_screenshots.mjs --password-file data\fixture_stack\user_password.txt`. The model cards and PR curves come from `scripts\export_pr_curves.py SNAPSHOT`, then `scripts\sync_model_versions.py`.

After an API change, refresh the types: `.venv\Scripts\python scripts\export_openapi.py`, then `npm run types` in `frontend/`. `scripts\export_openapi.py --check` fails if the schema file is out of date.

To try it against recorded data with no `.env`, run `.venv\Scripts\python scripts\fixture_stack.py`, which leaves the API at `http://127.0.0.1:8765` with Explain switched off, and build with `API_ORIGIN` set to that. `--down` stops it.

### Building the dataset

The backfill can stop and resume safely. `WIKIMEDIA_USER_AGENT` and `USER_HASH_KEY` must be set in `.env`:

```powershell
.venv\Scripts\python scripts\backfill.py --target 2000 --out data\backfill_pilot   # about 15 minutes
.venv\Scripts\python scripts\backfill.py                                           # about 10 hours
.venv\Scripts\python scripts\check_backfill.py data\backfill
.venv\Scripts\python scripts\measure_capacity.py data\backfill                    # needs TEST_DATABASE_URL
.venv\Scripts\python scripts\plot_backfill.py data\backfill docs\images --browser "C:\Program Files\Google\Chrome\Application\chrome.exe"
```

Keeping edits for 3 days uses a projected 185 MB, or 37% of Supabase's free limit, at about 86,600 edits a day.

### Dataset snapshots

A snapshot is the exact data the models train on: `edits.parquet` and a `manifest.json` that records its time windows, counts, exclusions, and content hash. Git tracks a small `.dvc` pointer, and [DVC](https://dvc.org) stores the data on DagsHub. Training tools have their own environment, because MLflow's pins conflict with pre-commit's.

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

**If Wikipedia hides a revision**, the public copy has to change too. `jobs/purge/snapshots.py` does the recheck, rewrite, and push on a schedule (daily from the deletion log, `--full` weekly), for every snapshot and the golden set. You then commit and push the changed `.dvc` files, and its next run deletes the old version, if `SNAPSHOT_GC_ENABLED` is on in its own clone. By hand:

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

### Monitoring and continuous training

The monitor or the owner adds a `retrain_requests` row, and the poller runs `TRAINER_COMMAND` with the request's kind and id. In the cluster the monitor and the poller are CronJobs, and the poller starts each run as a Kubernetes Job from the chart's template (`TRAINER_JOB_TEMPLATE`). By hand:

```powershell
.venv\Scripts\python jobs\monitor\monitor.py                         # daily; Evidently needs the trainer's environment
.venv\Scripts\python jobs\retrain_poller\poller.py                   # with TRAINER_COMMAND=python -m trainer.lifecycle
.venv-train\Scripts\python -m trainer.lifecycle retrain --request 12 --run 4
.venv-train\Scripts\python -m trainer.storage                          # retention, report only; --apply deletes
.venv-train\Scripts\python -m trainer.reference snapshots\NAME --version 1 --model-folder data\runs\roberta\model
```

Settings: `RETRAIN_INTERVAL_DAYS` (1, 7, or 14), `RETRAIN_ON_DRIFT`, `RETRAIN_STARTUP_DELAY_SECONDS`, `RETRAIN_MODELS` (CPU models only), `RETRAIN_TIMEOUT_MINUTES`, `RETRAIN_THREADS`, `SHADOW_HOURS`, `SNAPSHOT_GC_ENABLED`, and `STORAGE_ALERT_BYTES`, all in `.env.example`. Migration 0004 adds `shadow_scores`, so run the migrations again.

To replay the loops in a sandbox, against a throwaway Postgres (never Supabase): `.venv-train\Scripts\python scripts\simulate_continuous_training.py drift --admin-url postgresql://USER:PASSWORD@127.0.0.1:5434/postgres`. The other scenarios are `poisoning`, `rollback`, and `registry_down` (a retrain with MLflow down). It reads no `.env`, and only reads `data/runs`.

### Load tests

`python scripts/load_test.py up` starts the API at Render's free limits (0.1 CPU, 512 MB) over the demo's dataset, then `predict`, `reads --mode normal|spike|soak [--pacing browser]`, `explain`, and `replay` each run one k6 or replay scenario and write `data/load_test/SCENARIO-LABEL.json`; `down` removes it all. It needs Docker and k6 (`winget install GrafanaLabs.k6`). Explain talks to a fake LLM (`tests/e2e/fake_llm.py`) over TLS with a certificate made for the run, so nothing calls a real one. `--set LT_PREDICT_TIMEOUT_MS 300` and `--label` make a tuning run to compare.

### Usage per visitor

`python scripts/measure_usage.py` (after `python scripts/load_test.py up`) builds the frontend as Vercel does, puts the real Worker in front, and has a guest in Chromium sit on each page while it counts requests and the API's and Postgres's bytes. Results in `data/usage.json`; what they mean for each free limit is summarised in the project's cost notes.

### Rehearsals and the log check

These run on the same local stack as the load tests (`python scripts/load_test.py up` first, `down` after) and read no `.env`:
- `python scripts/rehearse_incident.py` rotates every secret the API holds, then plays a leaked-password incident, following `docs/runbooks/secret-rotation.md` and `security-incident.md`, while a visitor polls through the real Worker. Results in `data/rehearsal.json`.
- `python scripts/rehearse_restore.py --admin-url postgresql://USER:PASSWORD@127.0.0.1:PORT/postgres` backs the stack's database up, rotates the backup key, and rebuilds the demo's database in a fresh throwaway Postgres (never Supabase), checking every table.
- `python scripts/check_logs.py browser` builds the frontend with secrets in its environment and checks in Chromium that nothing a browser receives or stores holds one. `python scripts/check_logs.py stack` runs on its own: the whole pipeline, then every log searched for secrets, tokens, usernames, and diff text.

### Penetration test

`python scripts/pentest.py` runs every attack in the plan's list against the load test's stack (the API at Render's limits on the demo's data) behind the real Worker, `edge/worker.js`, run in Node by `scripts/edge_local.mjs`. It records each attack, its OWASP API or LLM Top 10 category, and whether it held in `data/pentest.json`. `--only edge auth brute authz input llm seat other logs` runs some areas, `--browser` adds the XSS check in Chromium (`scripts/pentest_browser.mjs`), and `--skip-build` uses a frontend you've already built. It needs Docker, Node, and about 15 minutes, most of it the guest-seat attacks; `--browser` also needs the frontend's Playwright Chromium. It shortens guest sessions to 2 minutes and the LLM's daily cap to 40 for the test only, and calls nothing outside the laptop. Results: `docs/security.md`, section 7b.

### Failure injection

`python scripts/inject_failures.py` breaks each row of the failure table (plan, section 14) on the end-to-end Compose stack and records what happened in `data/failure_injection.json`: the stream, Wikimedia's rate limits, Redpanda, the champion, every model, Postgres, the registry, the LLM, the API, and the cluster half. `--rows 4 5` runs some, `--frontend` builds the app and screenshots each outage in Chromium (`scripts/capture_failure_states.mjs`), `--keep` leaves the stack up. It needs Docker and about an hour with `--frontend`, because two rows hold their outage past the 10-minute freshness threshold. It calls no outside service: Wikimedia is a mock with switches (`docker-compose.failures.yml`), and the LLM hosts and the registry point at addresses where nothing listens.

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

### Loading the demo's data

The public demo's database holds the snapshot's test window (see [Project status](#project-status)). Into a migrated database with no other edits, from the repo root:

```bash
python scripts/sync_model_versions.py --runs data/runs              # the leaderboard's rows (DATABASE_URL)
python scripts/load_demo.py snapshots/snapshot-20261104-28c825af4b19 --dry-run
python scripts/load_demo.py snapshots/snapshot-20261104-28c825af4b19   # DATABASE_MIGRATIONS_URL
```

It needs the backfill's parts in `data/backfill/parts` and the trained runs in `data/runs`, loads in one transaction, and adds nothing on a second run. It refuses a database that already holds other edits. Then run the API with `DATA_MODE=snapshot`. Nothing runs the purge job on a schedule without the cluster, so run `jobs/purge` against the same database before the demo goes public, and now and then after.

### CI, delivery, and releases

Workflows are in `.github/workflows/`. Every action is pinned to a commit, the token is read-only unless a job says otherwise, and CI holds no secrets, so a pull request from a fork runs the same checks with nothing to take. `tests/test_pipeline_rules.py` fails the build if a workflow, Dockerfile, or lock file breaks one of these rules.

| Workflow | When | What |
|---|---|---|
| `ci.yml` | Every pull request and push, and weekly | Lint and types; the Python suite on real Postgres and Redpanda, with coverage gates of 80% for the shared package and 70% for each service and job; the trainer's tests and a training smoke test on CPU PyTorch; the Compose end-to-end test; a build and Trivy scan of every image; the frontend checks; `pip-audit` and `npm audit`; gitleaks. A pull request runs only the jobs its files can affect |
| `codeql.yml` | Every pull request and push, and weekly | Code scanning for Python, TypeScript, and the workflows themselves |
| `cd.yml` | After CI passes on `main` | Builds every image for amd64 and arm64, scans it, and publishes it to GHCR. Then migrates, deploys, and smoke-tests staging, and repeats for production. For the cluster, opens a pull request that pins each image's digest, and Argo CD deploys it once merged |
| `security-scan.yml` | By hand, and before a release | OWASP ZAP baseline and Schemathesis API fuzzing against staging |
| `release.yml` | A `vX.Y.Z` tag | Gives the tested images the version name and publishes the changelog entry |

The deploy half of `cd.yml` stays off until the repository variable `DEPLOY_ENABLED` is `true`, and the cluster's pull request until `CLUSTER_ENABLED` is `true`. It needs two GitHub environments, `staging` and `production`, each with the secrets `DATABASE_MIGRATIONS_URL` and `RENDER_DEPLOY_HOOK_URL` and the variable `API_URL`.

To rehearse CI's heavier jobs locally:

```powershell
.venv-train\Scripts\python scripts\make_tiny_runs.py data	iny_runs       # training smoke test and ONNX parity
.venv\Scripts\python scripts\compose_e2e.py --runs data	iny_runs          # the end-to-end test, as CI runs it
.venv\Scripts\python scripts\smoke_test.py --api http://127.0.0.1:8765     # the deploy smoke test, on the fixture stack
API_URL=http://127.0.0.1:8765 bash scripts/scan_api.sh                     # the staging API scan, on the fixture stack
bash scripts/protect_main.sh                                               # shows the branch protection it would set
```

To release, move the notes under "Unreleased" in [CHANGELOG.md](CHANGELOG.md) to a new version heading, merge, wait for CD, then `git tag -a v0.6.0 -m "v0.6.0"` and `git push origin v0.6.0`.

### Deploying the frontend to Vercel

The frontend is deployed at [wiki-guard.vercel.app](https://wiki-guard.vercel.app) on the Hobby plan, and every push to `main` deploys again. To set it up from scratch:

1. In Vercel, import the GitHub repo, and set **Root Directory** to `frontend`. Vercel detects Next.js, and `package.json` asks for Node 22.
2. Add no environment variables. In production the Cloudflare Worker sends `/api` to Render, so Vercel never needs `API_ORIGIN`, and anything in a `NEXT_PUBLIC_` variable reaches every browser.
3. Deploy, then check one page's headers for `content-security-policy` with a nonce: `curl -sI https://wiki-guard.vercel.app/`.

Until the API is on Render, only the landing and privacy pages work there; the console says it can't reach the API.

**What the public demo shows.** The streaming pipeline can't run around the clock for free, so the demo serves a fixed set of real edits: the 25,294 English Wikipedia edits from 26 October to 2 November 2026 that form the models' test window, which no model trained on. Each carries the champion's stored test score and its real revert label, so the demo's numbers are the evaluation's numbers (PR-AUC 0.364). The API runs with `DATA_MODE=snapshot`, and every console page says the data is a snapshot, not live. `/predict` and Explain still run on request.

### The cluster, the API on Render, and the edge

Everything that runs in Kubernetes is in `k8s/`, and [k8s/README.md](k8s/README.md) has the steps: the chart, the Secrets each component needs, the VM, and what keeps it safe. To run the whole cluster half on the laptop with kind, against the live Wikimedia stream and with no account:

```bash
python scripts/kind_cluster.py up                                  # cluster, images, monitoring, chart
bash scripts/check_network_policies.sh --context kind-wikiguard    # 13 allowed and refused connections
kubectl --context kind-wikiguard -n monitoring port-forward svc/kps-grafana 13000:80
node scripts/capture_grafana.mjs --password-file data/kind/grafana_password.txt
python scripts/kind_cluster.py down
```

For a VM, `scripts/harden_vm.sh` and `scripts/bootstrap_cluster.sh` install k3s and Argo CD, which then pulls the rest from `main`. They were written for Oracle's free ARM VM and have never run on one: Oracle had no capacity.

The API runs on Render's free plan from [render.yaml](render.yaml), which describes production and an on-demand staging service. Its build downloads the champion and the MLP into the image (`PREFETCH_MODELS=true`), so a wake from sleep needs no download.

The single site address is a Cloudflare Worker ([edge/](edge/)): `/api` goes to Render and everything else to Vercel, so the session cookie stays first-party. Its tests use Node's own runner: `node --test edge/worker.test.mjs`. To deploy it, from `edge/`: `npx wrangler secret put EDGE_SHARED_SECRET` (the same value as the API's), then `npx wrangler deploy`.

## Design notes

The design notes are kept locally and aren't in this repo: the full plan and task list, a step-by-step walkthrough, the threat model and privacy record, 96 recorded decisions with the alternatives rejected, findings with their evidence, the evaluation method, and every measured result. Only the images this README uses are tracked, in `docs/images/readme/`.

## Data and attribution

WikiGuard uses public data from Wikimedia. Wikipedia content is available under [CC BY-SA](https://creativecommons.org/licenses/by-sa/4.0/). Dataset snapshots include that text, so the same license applies to them, and each manifest says so. A revision's authors are listed through `https://en.wikipedia.org/w/index.php?diff=REV_ID`. WikiGuard is an independent project and isn't affiliated with the Wikimedia Foundation. Test fixtures in [tests/fixtures/wikimedia/](tests/fixtures/wikimedia/README.md) are scrubbed recordings, and that folder's README covers what was changed and the attribution for kept text.
