# ruff: noqa: E501 - the queries are data, and each reads better on one line than split up
"""Write the five Grafana dashboards the chart ships (k8s/charts/wikiguard/dashboards/).

    python scripts/build_dashboards.py           # write them
    python scripts/build_dashboards.py --check   # fail if the files differ from what this writes

Dashboards are generated, not hand-edited, because Grafana's JSON is long and repetitive, and one
place for each query is easier to review. Two data sources:

- Prometheus, for everything the cluster runs.
- Postgres, through the read-only observer role (DEC-085), for what only the API records: sign-ins,
  denied actions, admin use, and LLM calls. Prometheus can't scrape the API on Render, because each
  scrape would keep it awake (F-015).
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

OUT = Path(__file__).resolve().parents[1] / "k8s/charts/wikiguard/dashboards"
PROM = {"type": "prometheus", "uid": "prometheus"}
PG = {"type": "grafana-postgresql-datasource", "uid": "wikiguard-postgres"}
NS = 'namespace="wikiguard"'
# One color per model, the same as the app's own charts (frontend/src/app/globals.css).
MODEL_COLORS = {
    "roberta": "#7c3aed",
    "distilbert": "#0891b2",
    "bilstm": "#65a30d",
    "bert": "#db2777",
    "mlp": "#64748b",
}


class Board:
    def __init__(self, uid: str, title: str, description: str) -> None:
        self.uid, self.title, self.description = uid, title, description
        self.panels: list[dict[str, Any]] = []
        self.y = 0
        self.x = 0
        self.row_height = 0

    def _place(self, width: int, height: int) -> dict[str, int]:
        if self.x + width > 24:
            self.x, self.y = 0, self.y + self.row_height
            self.row_height = 0
        position = {"x": self.x, "y": self.y, "w": width, "h": height}
        self.x += width
        self.row_height = max(self.row_height, height)
        return position

    def row(self, title: str) -> None:
        if self.x:
            self.x, self.y = 0, self.y + self.row_height
        self.panels.append(
            {
                "type": "row",
                "title": title,
                "collapsed": False,
                "gridPos": {"x": 0, "y": self.y, "w": 24, "h": 1},
                "id": len(self.panels) + 1,
                "panels": [],
            }
        )
        self.y += 1
        self.row_height = 0

    def add(self, panel: dict[str, Any], width: int, height: int) -> None:
        panel["id"] = len(self.panels) + 1
        panel["gridPos"] = self._place(width, height)
        self.panels.append(panel)

    def stat(self, title: str, expr: str, unit: str, description: str, **options: Any) -> None:
        thresholds = options.get("thresholds") or [{"color": "text", "value": None}]
        self.add(
            {
                "type": "stat",
                "title": title,
                "description": description,
                "datasource": PROM,
                "targets": [{"refId": "A", "expr": expr, "datasource": PROM}],
                "fieldConfig": {
                    "defaults": {
                        "unit": unit,
                        "decimals": options.get("decimals"),
                        "thresholds": {"mode": "absolute", "steps": thresholds},
                        "color": {"mode": "thresholds"},
                    },
                    "overrides": [],
                },
                "options": {
                    "reduceOptions": {"calcs": ["lastNotNull"], "fields": "", "values": False},
                    "colorMode": "value",
                    "graphMode": "area",
                    "textMode": "value_and_name" if options.get("named") else "value",
                },
            },
            options.get("width", 6),
            4,
        )

    def series(
        self,
        title: str,
        targets: list[tuple[str, str]],
        unit: str,
        description: str,
        width: int = 12,
        height: int = 8,
        threshold: float | None = None,
        source: dict[str, str] = PROM,
        colors: dict[str, str] | None = None,
        stack: bool = False,
    ) -> None:
        overrides = [
            {
                "matcher": {"id": "byName", "options": name},
                "properties": [{"id": "color", "value": {"mode": "fixed", "fixedColor": color}}],
            }
            for name, color in (colors or {}).items()
        ]
        defaults: dict[str, Any] = {
            "unit": unit,
            "custom": {
                "drawStyle": "line",
                "lineWidth": 2,
                "fillOpacity": 25 if stack else 10,
                "showPoints": "never",
                "spanNulls": True,
                "stacking": {"mode": "normal" if stack else "none"},
                "thresholdsStyle": {"mode": "line+area" if threshold is not None else "off"},
            },
            "color": {"mode": "palette-classic"},
            "thresholds": {
                "mode": "absolute",
                "steps": [{"color": "transparent", "value": None}]
                + ([{"color": "red", "value": threshold}] if threshold is not None else []),
            },
        }
        self.add(
            {
                "type": "timeseries",
                "title": title,
                "description": description,
                "datasource": source,
                "targets": [
                    self._target(source, ref, query, legend)
                    for ref, (query, legend) in zip("ABCDEFGH", targets, strict=False)
                ],
                "fieldConfig": {"defaults": defaults, "overrides": overrides},
                "options": {
                    "legend": {
                        "displayMode": "table",
                        "placement": "bottom",
                        "calcs": ["lastNotNull", "max"],
                    },
                    "tooltip": {"mode": "multi", "sort": "desc"},
                },
            },
            width,
            height,
        )

    def _target(self, source: dict[str, str], ref: str, query: str, legend: str) -> dict[str, Any]:
        if source is PG:
            return {
                "refId": ref,
                "rawSql": query,
                "format": "time_series",
                "rawQuery": True,
                "editorMode": "code",
                "datasource": PG,
            }
        return {"refId": ref, "expr": query, "legendFormat": legend, "datasource": PROM}

    def table(
        self, title: str, sql: str, description: str, width: int = 12, height: int = 8
    ) -> None:
        self.add(
            {
                "type": "table",
                "title": title,
                "description": description,
                "datasource": PG,
                "targets": [
                    {
                        "refId": "A",
                        "rawSql": sql,
                        "format": "table",
                        "rawQuery": True,
                        "editorMode": "code",
                        "datasource": PG,
                    }
                ],
                "options": {"showHeader": True, "cellHeight": "sm"},
                "fieldConfig": {"defaults": {}, "overrides": []},
            },
            width,
            height,
        )

    def pg_stat(
        self,
        title: str,
        sql: str,
        unit: str,
        description: str,
        steps: list[dict[str, Any]],
        width: int = 6,
    ) -> None:
        self.add(
            {
                "type": "stat",
                "title": title,
                "description": description,
                "datasource": PG,
                "targets": [
                    {
                        "refId": "A",
                        "rawSql": sql,
                        "format": "table",
                        "rawQuery": True,
                        "editorMode": "code",
                        "datasource": PG,
                    }
                ],
                "fieldConfig": {
                    "defaults": {
                        "unit": unit,
                        "thresholds": {"mode": "absolute", "steps": steps},
                        "color": {"mode": "thresholds"},
                    },
                    "overrides": [],
                },
                "options": {
                    "reduceOptions": {"calcs": ["lastNotNull"], "fields": "", "values": False},
                    "colorMode": "value",
                    "graphMode": "none",
                    "textMode": "value",
                },
            },
            width,
            4,
        )

    def text(self, title: str, markdown: str, width: int = 24, height: int = 6) -> None:
        self.add(
            {"type": "text", "title": title, "options": {"mode": "markdown", "content": markdown}},
            width,
            height,
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "uid": self.uid,
            "title": self.title,
            "description": self.description,
            "tags": ["wikiguard"],
            "timezone": "utc",
            "schemaVersion": 39,
            "version": 1,
            "editable": False,
            "graphTooltip": 1,
            "refresh": "1m",
            "time": {"from": "now-6h", "to": "now"},
            "panels": self.panels,
            "templating": {"list": []},
            "annotations": {"list": []},
            "links": [],
        }


GREEN_AMBER_RED = [
    {"color": "green", "value": None},
    {"color": "orange", "value": 0.7},
    {"color": "red", "value": 0.85},
]


def pipeline() -> Board:
    board = Board(
        "wikiguard-pipeline",
        "WikiGuard / Pipeline",
        "Wikipedia's stream to a stored score: ingest, score, archive. Section 11 of the plan.",
    )
    board.row("Right now")
    board.stat(
        "Edits published per minute",
        "sum(rate(ingestor_published_total[5m])) * 60",
        "short",
        "In-scope edits the ingestor sent to edits.raw. English Wikipedia's articles see "
        "about one human edit a second.",
        decimals=0,
    )
    board.stat(
        "Edit to score, p95",
        "histogram_quantile(0.95, sum by (le) (rate(scorer_edit_to_score_seconds_bucket[10m])))",
        "s",
        "From the edit's timestamp on Wikipedia to its stored score. SLO: 60 s.",
        thresholds=[
            {"color": "green", "value": None},
            {"color": "orange", "value": 60},
            {"color": "red", "value": 300},
        ],
    )
    board.stat(
        "Scorer backlog",
        "max(scorer_consumer_lag)",
        "short",
        "Edits waiting in edits.raw. Past 500 the scorer uses only the fast models.",
        thresholds=[
            {"color": "green", "value": None},
            {"color": "orange", "value": 250},
            {"color": "red", "value": 500},
        ],
        decimals=0,
    )
    board.stat(
        "Fallback rate, 15 min",
        'sum(rate(scorer_edits_scored_total{fallback="true"}[15m])) / '
        "sum(rate(scorer_edits_scored_total[15m]))",
        "percentunit",
        "Share of edits scored by a model behind the champion. Alert: 5%.",
        thresholds=[
            {"color": "green", "value": None},
            {"color": "orange", "value": 0.02},
            {"color": "red", "value": 0.05},
        ],
        decimals=1,
    )
    board.row("Flow")
    board.series(
        "Edits per minute",
        [
            ("sum(rate(ingestor_published_total[5m])) * 60", "published"),
            ("sum(rate(scorer_edits_scored_total[5m])) * 60", "scored"),
            (
                "sum by (dataset) (rate(archiver_rows_written_total[5m])) * 60",
                "archived {{dataset}}",
            ),
        ],
        "short",
        "Each stage should track the one before it.",
    )
    board.series(
        "Edit to score",
        [
            (
                "histogram_quantile(0.5, sum by (le) (rate(scorer_edit_to_score_seconds_bucket[10m])))",
                "p50",
            ),
            (
                "histogram_quantile(0.95, sum by (le) (rate(scorer_edit_to_score_seconds_bucket[10m])))",
                "p95",
            ),
        ],
        "s",
        "The red band starts at the 60-second SLO (docs/slo.md).",
        threshold=60,
    )
    board.series(
        "Scorer backlog",
        [("max(scorer_consumer_lag)", "edits behind")],
        "short",
        "Load shedding starts at 500 and stops at 250.",
        threshold=500,
    )
    board.series(
        "Stream health",
        [
            ("max(ingestor_lag_seconds)", "age of newest edit"),
            ("max(ingestor_last_gap_seconds)", "largest gap between events"),
        ],
        "s",
        "A growing age or a long gap means the stream is slow or reconnecting.",
    )
    board.row("Failures")
    board.series(
        "Dead letters, per hour",
        [
            ("sum by (reason) (increase(ingestor_dead_letters_total[1h]))", "ingestor: {{reason}}"),
            ("sum by (reason) (increase(scorer_dlq_total[1h]))", "scorer: {{reason}}"),
            ("sum(increase(archiver_poison_messages_total[1h]))", "archiver: unreadable"),
        ],
        "short",
        "Events that couldn't be handled. They go to edits.dlq, not into a retry loop.",
    )
    board.series(
        "Reconnects and retries, per hour",
        [
            ("sum(increase(ingestor_reconnects_total[1h]))", "stream reconnects"),
            ("sum(increase(scorer_delivery_retries_total[1h]))", "scorer write retries"),
            ("sum(increase(scorer_stored_unscored_total[1h]))", "stored unscored"),
            ("sum(increase(scorer_rescored_total[1h]))", "rescored later"),
        ],
        "short",
        "Recovered failures. A steady climb is worth a look before it becomes an outage.",
    )
    board.series(
        "Pod restarts",
        [
            (
                f"sum by (pod) (increase(kube_pod_container_status_restarts_total{{{NS}}}[1h]))",
                "{{pod}}",
            ),
        ],
        "short",
        "Restarts per hour in the wikiguard namespace.",
        width=24,
        height=7,
    )
    return board


def models() -> Board:
    board = Board(
        "wikiguard-models",
        "WikiGuard / Models",
        "The serving chain in the cluster: which model answered, how fast, and how often it fell back.",
    )
    board.row("Which model answered")
    board.series(
        "Edits scored per minute, by model",
        [
            ("sum by (model) (rate(scorer_edits_scored_total[5m])) * 60", "{{model}}"),
        ],
        "short",
        "The champion should answer almost every edit. Others mean a fallback.",
        colors=MODEL_COLORS,
        stack=True,
    )
    board.series(
        "Fallback and load shedding",
        [
            (
                'sum(rate(scorer_edits_scored_total{fallback="true"}[15m])) / '
                "sum(rate(scorer_edits_scored_total[15m]))",
                "fallback rate",
            ),
            ("max(scorer_load_shedding)", "load shedding (1 = on)"),
        ],
        "percentunit",
        "The red band starts at the 5% fallback alert.",
        threshold=0.05,
    )
    board.row("Speed and failures")
    board.series(
        "Batch latency, p95, by model",
        [
            (
                "histogram_quantile(0.95, sum by (le, model) (rate(scorer_batch_latency_seconds_bucket[10m])))",
                "{{model}}",
            ),
        ],
        "s",
        "Per micro-batch: Kafka read, features, model, and the Postgres write.",
        colors=MODEL_COLORS,
    )
    board.series(
        "Circuit breakers",
        [("max by (model) (scorer_breaker_open)", "{{model}}")],
        "short",
        "1 while a model is skipped after 3 failures in a row (DEC-054).",
        colors=MODEL_COLORS,
    )
    board.series(
        "Shadow candidate",
        [
            (
                "sum by (outcome) (rate(scorer_shadow_scored_total[15m])) * 60",
                "{{outcome}} per minute",
            ),
            ("max(scorer_shadow_loaded)", "candidate loaded"),
        ],
        "short",
        "During a 24-hour shadow period, the candidate scores live edits with no effect "
        "on users (section 4).",
    )
    board.series(
        "Alias reload failures, per hour",
        [
            ("sum(increase(scorer_reload_failures_total[1h]))", "failed reloads"),
        ],
        "short",
        "A failed reload keeps the old model serving. Repeated failures need a look.",
    )
    board.row("Registry")
    board.table(
        "Serving versions",
        "SELECT model_name AS model, model_version AS version, alias, registered_at "
        "FROM model_versions WHERE alias IS NOT NULL ORDER BY alias",
        "What the registry says serves, as the trainer last recorded it.",
    )
    board.table(
        "Recent retraining",
        "SELECT id, trigger, status, started_at, finished_at FROM retrain_runs "
        "ORDER BY id DESC LIMIT 10",
        "Each retrain and its outcome. The Health page in the app shows the same.",
    )
    return board


def api() -> Board:
    board = Board(
        "wikiguard-api",
        "WikiGuard / API",
        "What the API on Render records in Postgres. Its /metrics isn't scraped, because each "
        "scrape would keep the free service awake (F-015).",
    )
    board.text(
        "Where the rest is",
        (
            "Per-route latency and error rates are on the API's own `/metrics`, which needs a bearer "
            "token and isn't scraped from here: a scrape every few minutes would keep Render's free "
            "service awake around the clock and use up the 750 shared hours (F-015). The load tests in "
            "Phase 8 read it directly. Render's dashboard shows requests and memory too."
        ),
        height=4,
    )
    board.series(
        "Sign-ins per hour, by kind",
        [
            (
                "SELECT date_trunc('hour', at) AS time, action || ' ' || actor AS metric, count(*) AS value "
                "FROM audit_log WHERE $__timeFilter(at) AND action LIKE 'auth.%' "
                "GROUP BY 1, 2 ORDER BY 1",
                "",
            ),
        ],
        "short",
        "From the audit log: guest and user sign-ins, sign-outs, and failures.",
        source=PG,
    )
    board.series(
        "LLM calls per day, by provider",
        [
            (
                "SELECT day::timestamptz AS time, provider AS metric, calls AS value FROM llm_usage "
                "WHERE $__timeFilter(day::timestamptz) ORDER BY 1",
                "",
            ),
        ],
        "short",
        "The durable counter behind Explain's daily cap (section 7).",
        source=PG,
    )
    board.series(
        "Seat take-overs and admin actions per hour",
        [
            (
                "SELECT date_trunc('hour', at) AS time, action AS metric, count(*) AS value FROM audit_log "
                "WHERE $__timeFilter(at) AND (action LIKE 'gate.%' OR action LIKE 'admin.%') "
                "GROUP BY 1, 2 ORDER BY 1",
                "",
            ),
        ],
        "short",
        "Only the user account can do these.",
        source=PG,
    )
    board.series(
        "Retrain requests per day, by status",
        [
            (
                "SELECT date_trunc('day', requested_at) AS time, kind || ' ' || status AS metric, "
                "count(*) AS value FROM retrain_requests WHERE $__timeFilter(requested_at) "
                "GROUP BY 1, 2 ORDER BY 1",
                "",
            ),
        ],
        "short",
        "The API and the monitor ask; only the poller in the cluster acts (DEC-071).",
        source=PG,
    )
    return board


def security() -> Board:
    board = Board(
        "wikiguard-security",
        "WikiGuard / Security",
        "Section 8's detect-and-respond signals: failed sign-ins, denied actions, admin use, "
        "LLM spend, and the cluster's own health.",
    )
    board.row("Access")
    board.pg_stat(
        "Failed sign-ins, 24 h",
        "SELECT count(*) FROM audit_log WHERE action = 'auth.sign_in_failed' "
        "AND at > now() - interval '24 hours'",
        "short",
        "Wrong password for the user account. Lockout starts after a few in a row.",
        [
            {"color": "green", "value": None},
            {"color": "orange", "value": 10},
            {"color": "red", "value": 50},
        ],
    )
    board.pg_stat(
        "Denied actions, 24 h",
        "SELECT count(*) FROM audit_log WHERE action = 'auth.denied' "
        "AND at > now() - interval '24 hours'",
        "short",
        "A guest or a signed-out caller asked for something only user may do.",
        [
            {"color": "green", "value": None},
            {"color": "orange", "value": 10},
            {"color": "red", "value": 50},
        ],
    )
    board.pg_stat(
        "Admin actions, 24 h",
        "SELECT count(*) FROM audit_log WHERE action LIKE 'admin.%' "
        "AND at > now() - interval '24 hours'",
        "short",
        "Retrain, promote, rollback, and overrides. Expected only from the owner.",
        [{"color": "blue", "value": None}],
    )
    board.pg_stat(
        "LLM calls today",
        "SELECT coalesce(sum(calls), 0) FROM llm_usage WHERE day = current_date",
        "short",
        "Against the daily cap (EXPLAIN_DAILY_CALL_CAP). Alerts at 50% and 80%.",
        [{"color": "blue", "value": None}],
    )
    board.series(
        "Failed sign-ins and denied actions per 15 min",
        [
            (
                "SELECT $__timeGroupAlias(at, '15m'), action AS metric, count(*) AS value FROM audit_log "
                "WHERE $__timeFilter(at) AND action IN ('auth.sign_in_failed', 'auth.denied') "
                "GROUP BY 1, 2 ORDER BY 1",
                "",
            ),
        ],
        "short",
        "A spike is someone trying passwords or calling user-only routes.",
        source=PG,
    )
    board.table(
        "Latest admin and gate actions",
        "SELECT at, actor, action, target FROM audit_log "
        "WHERE action LIKE 'admin.%' OR action LIKE 'gate.%' ORDER BY id DESC LIMIT 20",
        "The audit log is append-only for every runtime role (section 8).",
    )
    board.row("Cluster")
    board.series(
        "Container restarts per hour",
        [
            (
                f"sum by (container) (increase(kube_pod_container_status_restarts_total{{{NS}}}[1h]))",
                "{{container}}",
            ),
        ],
        "short",
        "A crash loop can be an attack on the parser, or a bad deploy.",
    )
    board.series(
        "Pods not ready",
        [
            # Only running pods: a finished job's pod also reports "not ready", and isn't a problem.
            (
                f'sum by (pod) (kube_pod_status_ready{{{NS},condition="false"}} * on (namespace, pod) '
                f'kube_pod_status_phase{{{NS},phase="Running"}})',
                "{{pod}}",
            ),
        ],
        "short",
        "A running pod failing its readiness probe. Anything above zero for long is down.",
    )
    return board


def free_tier() -> Board:
    board = Board(
        "wikiguard-free-tier",
        "WikiGuard / Free-tier usage",
        "Every limit WikiGuard can measure from inside the cluster, against its free allowance "
        "(docs/cost.md). Nothing may start billing (DEC-001).",
    )
    board.row("Oracle VM (Always Free A1)")
    board.series(
        "CPU busy, and Oracle's idle line",
        [
            ('1 - avg(rate(node_cpu_seconds_total{mode="idle"}[5m]))', "CPU busy"),
        ],
        "percentunit",
        "Oracle may reclaim an A1 VM whose 95th-percentile CPU and memory both stay "
        "under 20% for 7 days. The band shows 20%.",
        threshold=0.2,
        width=8,
    )
    board.series(
        "Memory used",
        [
            (
                "1 - sum(node_memory_MemAvailable_bytes) / sum(node_memory_MemTotal_bytes)",
                "memory used",
            ),
        ],
        "percentunit",
        "Under 20% for a week counts toward reclaim, as CPU does.",
        threshold=0.2,
        width=8,
    )
    board.series(
        "Disk used",
        [
            (
                '1 - sum(node_filesystem_avail_bytes{mountpoint="/",fstype!~"tmpfs|overlay"}) / '
                'sum(node_filesystem_size_bytes{mountpoint="/",fstype!~"tmpfs|overlay"})',
                "disk used",
            ),
        ],
        "percentunit",
        "Alert at 80%. The boot volume is part of the 200 GB free block storage.",
        threshold=0.8,
        width=8,
    )
    board.row("Storage")
    board.stat(
        "Archive, of Oracle's 20 GB",
        "max(archiver_archive_bytes) / (20 * 1024^3)",
        "percentunit",
        "The working copy, which the nightly mirror copies to Object Storage (DEC-083).",
        thresholds=[
            {"color": "green", "value": None},
            {"color": "orange", "value": 0.7},
            {"color": "red", "value": 0.8},
        ],
        width=8,
        decimals=1,
    )
    board.pg_stat(
        "Supabase database, of 500 MB",
        "SELECT pg_database_size(current_database()) / (500.0 * 1024 * 1024)",
        "percentunit",
        "The retention job shortens the windows at 85% and 95%.",
        GREEN_AMBER_RED,
        width=8,
    )
    board.stat(
        "Edits per day",
        "sum(increase(ingestor_published_total[24h]))",
        "short",
        "What drives Supabase's size and the archive's growth (docs/capacity.md).",
        width=8,
        decimals=0,
    )
    board.row("Calls")
    board.series(
        "LLM calls per day",
        [
            (
                "SELECT day::timestamptz AS time, provider AS metric, calls AS value FROM llm_usage "
                "WHERE $__timeFilter(day::timestamptz) ORDER BY 1",
                "",
            ),
        ],
        "short",
        "OpenAI is prepaid and Gemini is free. Both stop at the daily cap.",
        source=PG,
    )
    board.text(
        "Checked by hand",
        (
            "These limits have no API this cluster can read for free, so they're read from each "
            "dashboard and recorded in `docs/cost.md`:\n\n"
            "| Service | Free limit | Where to read it |\n|---|---|---|\n"
            "| Render | 750 instance hours a month, shared | Render dashboard, Billing |\n"
            "| Vercel Hobby | Function and bandwidth allowances | Vercel dashboard, Usage |\n"
            "| Cloudflare Workers | 100,000 requests a day | Cloudflare dashboard, Workers |\n"
            "| Supabase | 5 GB egress a month | Supabase dashboard, Usage |\n"
            "| Oracle Object Storage | 50,000 requests a month | Oracle console, Cost Analysis |\n"
            "| DagsHub | Storage (F-071) | DagsHub repo settings |"
        ),
        width=12,
        height=9,
    )
    return board


BOARDS = (pipeline, models, api, security, free_tier)


def render() -> dict[str, str]:
    return {
        f"{board.uid.removeprefix('wikiguard-')}.json": json.dumps(board.to_json(), indent=2) + "\n"
        for board in (make() for make in BOARDS)
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    files = render()
    if args.check:
        stale = [
            name
            for name, text in files.items()
            if not (OUT / name).is_file() or (OUT / name).read_text("utf-8") != text
        ]
        for name in stale:
            print(f"{name} is out of date: run scripts/build_dashboards.py", file=sys.stderr)  # noqa: T201
        return 1 if stale else 0
    OUT.mkdir(parents=True, exist_ok=True)
    for name, text in files.items():
        (OUT / name).write_text(text, "utf-8", newline="\n")
    print(f"wrote {len(files)} dashboards to {OUT}")  # noqa: T201
    return 0


if __name__ == "__main__":
    sys.exit(main())
