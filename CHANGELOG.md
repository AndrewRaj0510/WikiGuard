# Changelog

What changed in each version of WikiGuard. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Until 1.0.0 the minor number is the build phase that the version completes, so 0.6.0 is the end
of Phase 6. 1.0.0 is the launch, after production hardening. Versions before 0.6.0 were written up
afterwards from the commit that finished each phase, and the links below point at those commits.

To release: move the notes under "Unreleased" to a new version heading, merge, wait for the CD
workflow to publish that commit's images, then push a `vX.Y.Z` tag. The release workflow gives the
tested images the version name and publishes these notes.

## [Unreleased]

### Added
- Full CI on every pull request: lint, types, the whole Python suite on real Postgres and
  Redpanda with a coverage gate for each part, the trainer's tests and a training smoke test on
  CPU PyTorch, a Compose end-to-end run, an image build and Trivy scan for every image, the
  frontend checks, `pip-audit`, `npm audit`, gitleaks, and CodeQL.
- Path filters, so a pull request runs only the jobs its files can affect, and one `ci` check for
  branch protection to require.
- Hash-locked Python installs. Each `requirements.in` compiles to a `requirements.txt` that pins
  every package with its SHA-256 (`scripts/lock_requirements.py`), and images and CI install with
  `--require-hashes`.
- A CD workflow: after CI passes on `main`, every image is built for amd64 and arm64 on native
  runners, scanned, and published to GHCR. It then migrates and deploys staging, smoke-tests it,
  and repeats for production. The deploy half stays off until Phase 7 creates the services.
- A deploy smoke test (`scripts/smoke_test.py`) that waits for the new build and checks it's
  still locked. `/health` now reports the commit it was built from.
- API fuzzing from the OpenAPI schema with Schemathesis, in the test suite and as a scan of
  staging, next to an OWASP ZAP baseline scan.
- A test that keeps migrations to expand-then-contract, so the running code never breaks
  during a deploy.
- A release workflow, this changelog, and Dependabot for Actions, base images, Python, and npm.
- `/robots.txt`, which closes `/api/` to crawlers.

### Changed
- Every Dockerfile is now two stages on a base image pinned by digest. Compose images are pinned
  by digest too.
- The published API schema now describes validation errors as `application/problem+json`, which
  is what the API always sent. The frontend's generated types changed to match.
- A `.dockerignore` keeps everything except the code out of image builds.

### Fixed
- `/favicon.ico` and `/robots.txt` answered with a 404 page that had no Content-Security-Policy. Found by OWASP ZAP.

### Security
- Workflows hold no secrets on pull requests, use a read-only token by default, and pin every
  action to a commit. Tests in the suite fail if a workflow, Dockerfile, or lock file breaks
  these rules.

## [0.5.0] - 2026-11-11

### Added
- The monitor: live accuracy from revert labels, Evidently drift reports, and a retrain trigger.
- The trainer pipeline: snapshot, validation, training, evaluation, registration, a promotion
  gate, a shadow period, and a post-promotion check with rollback.
- A purge of hidden revisions from kept snapshots, and retention for model versions and
  snapshots.
- Replays that show the loop working on recorded data: drift to promotion, and a poisoned
  snapshot rejected.

## [0.4.0] - 2026-11-09

### Added
- The Next.js frontend: landing page, live feed, review queue, edit detail with Explain, model
  leaderboard, and model health.
- A nonce-based content security policy on every page, an XSS payload corpus run through the
  real UI, axe accessibility checks, and Lighthouse budgets in CI.
- The single-seat gate in the UI, with `guest` and `user` accounts.

## [0.3.0] - 2026-11-08

### Added
- The scorer, with a fallback chain that keeps scoring when a model fails.
- The FastAPI service: review queue, edit detail, leaderboard, health, `/predict`, and
  `/explain` with prompt-injection defenses.
- A retrain-request poller, so the API never holds cluster credentials.
- A Compose end-to-end test of the whole pipeline.

## [0.2.0] - 2026-11-07

### Added
- Versioned dataset snapshots with a content hash, tracked with DVC.
- Five models (MLP, BiLSTM, DistilBERT, BERT, RoBERTa) trained on one snapshot with time-based
  splits, exported to ONNX with int8 quantization, and registered in MLflow.
- Evaluation with confidence intervals, calibration, slices by editor type, a golden set, and a
  robustness report. RoBERTa became champion with a test PR-AUC of 0.3641.

## [0.1.0] - 2026-11-05

### Added
- The ingestor, reading English Wikipedia's edit stream into Redpanda, and the labeler, which
  collects revert labels after 48 hours.
- The archiver, which writes the stream to parquet, and the purge and retention jobs.
- The database schema with least-privilege roles and row level security on every table.
- A 30-day backfill, and the first CI.

[Unreleased]: https://github.com/AndrewRaj0510/WikiGuard/compare/ab2835a...HEAD
[0.5.0]: https://github.com/AndrewRaj0510/WikiGuard/compare/d3467ff...ab2835a
[0.4.0]: https://github.com/AndrewRaj0510/WikiGuard/compare/06adbd1...d3467ff
[0.3.0]: https://github.com/AndrewRaj0510/WikiGuard/compare/475af67...06adbd1
[0.2.0]: https://github.com/AndrewRaj0510/WikiGuard/compare/4eb8a84...475af67
[0.1.0]: https://github.com/AndrewRaj0510/WikiGuard/commit/4eb8a84
