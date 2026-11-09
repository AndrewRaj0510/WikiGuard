/**
 * Reading the Health summary's free-form parts. live_metrics, drift_reports, and retrain_runs are
 * written by the monitor and trainer (Phase 5), so each field is checked before use, and anything
 * malformed is skipped instead of breaking the page.
 */
import type { HealthSummary } from "./api/types";

export interface LiveMetric {
  day: string;
  modelVersion: string;
  prAuc: number | null;
  labeled: number | null;
}

export interface DriftReport {
  createdAt: string;
  drifted: boolean;
}

/** One side of a before vs after report. */
export interface Side {
  model: string | null;
  version: string | null;
  prAuc: number | null;
  prAucInterval: [number, number] | null;
}

/**
 * The trainer's comparison, in retrain_runs.details. This is the contract the Phase 5 trainer
 * writes (DEC-066): `before` is the serving champion, `after` the candidate, and `difference` the
 * paired bootstrap of after minus before on the same rows.
 */
export interface Comparison {
  before: Side;
  after: Side;
  difference: { mean: number; low: number; high: number } | null;
  promoted: boolean | null;
  reason: string | null;
}

export interface RetrainRun {
  id: number | null;
  trigger: string;
  status: string;
  startedAt: string;
  finishedAt: string | null;
  comparison: Comparison | null;
}

type Json = Record<string, unknown>;

const num = (v: unknown) => (typeof v === "number" && Number.isFinite(v) ? v : null);
const str = (v: unknown) => (typeof v === "string" && v ? v : null);
const obj = (v: unknown): Json =>
  v !== null && typeof v === "object" && !Array.isArray(v) ? (v as Json) : {};

function side(value: unknown): Side {
  const raw = obj(value);
  const interval = Array.isArray(raw.pr_auc_interval) ? raw.pr_auc_interval.map(num) : [];
  return {
    model: str(raw.model),
    version: raw.version === undefined || raw.version === null ? null : String(raw.version),
    prAuc: num(raw.pr_auc),
    prAucInterval:
      interval.length === 2 && interval[0] !== null && interval[1] !== null
        ? [interval[0], interval[1]]
        : null,
  };
}

export function comparisonOf(details: unknown): Comparison | null {
  const raw = obj(details);
  if (!("before" in raw) || !("after" in raw)) return null;
  const diff = obj(raw.difference);
  const mean = num(diff.mean);
  const low = num(diff.low);
  const high = num(diff.high);
  return {
    before: side(raw.before),
    after: side(raw.after),
    difference: mean !== null && low !== null && high !== null ? { mean, low, high } : null,
    promoted: typeof raw.promoted === "boolean" ? raw.promoted : null,
    reason: str(raw.reason),
  };
}

export function liveMetricsOf(summary: HealthSummary): LiveMetric[] {
  return summary.live_metrics
    .map((row) => {
      const metrics = obj(row.metrics);
      return {
        day: String(row.day ?? ""),
        modelVersion: String(row.model_version ?? ""),
        prAuc: num(metrics.pr_auc),
        labeled: num(metrics.labeled),
      };
    })
    .filter((row) => row.day)
    .sort((a, b) => a.day.localeCompare(b.day));
}

export function driftReportsOf(summary: HealthSummary): DriftReport[] {
  return summary.drift_reports
    .filter((row) => typeof row.created_at === "string" && typeof row.drifted === "boolean")
    .map((row) => ({ createdAt: row.created_at as string, drifted: row.drifted as boolean }));
}

export function retrainRunsOf(summary: HealthSummary): RetrainRun[] {
  return summary.retrain_runs
    .filter((row) => typeof row.started_at === "string")
    .map((row) => ({
      id: num(row.id),
      trigger: str(row.trigger) ?? "unknown",
      status: str(row.status) ?? "unknown",
      startedAt: row.started_at as string,
      finishedAt: str(row.finished_at),
      comparison: comparisonOf(row.details),
    }));
}

export type HealthTone = "ok" | "warn" | "danger";

export interface HealthSignal {
  tone: HealthTone;
  title: string;
  detail: string;
}

// The API caches the summary for 60 s, so polling faster would only re-read its cache.
export const HEALTH_POLL_MS = 60_000;

// A fallback answering this share of a day's edits means the champion is struggling.
export const FALLBACK_WARN_RATE = 0.1;
// With edits arriving about once a second, ten quiet minutes means the scorer has stopped.
export const STALE_SCORING_MINUTES = 10;

/**
 * What, if anything, the banner should say. Worst first. Kept apart from React so it's tested
 * against every case.
 */
export function healthSignals(summary: HealthSummary, now: number): HealthSignal[] {
  const signals: HealthSignal[] = [];
  const open = summary.chain.filter((item) => item.breaker !== "closed");
  if (summary.models_loaded && open.length) {
    signals.push({
      tone: "danger",
      title: "Degraded",
      detail: `${open.map((item) => item.model_name).join(", ")} ${open.length > 1 ? "are" : "is"} failing, so a fallback model is answering.`,
    });
  }
  if (summary.fallback_rate_24h !== null && summary.fallback_rate_24h >= FALLBACK_WARN_RATE) {
    signals.push({
      tone: "warn",
      title: "Degraded",
      detail: `${Math.round(summary.fallback_rate_24h * 100)}% of the last day’s edits were scored by a fallback model.`,
    });
  }
  if (summary.last_scored_at && summary.scored_last_24h > 0) {
    const minutes = (now - Date.parse(summary.last_scored_at)) / 60_000;
    if (minutes >= STALE_SCORING_MINUTES) {
      signals.push({
        tone: "warn",
        title: "Stale",
        detail: `No edit has been scored for ${Math.round(minutes)} minutes. The scorer may be stopped.`,
      });
    }
  }
  if (summary.stale) {
    signals.push({
      tone: "warn",
      title: "Showing cached data",
      detail: "The database didn’t answer, so these numbers may be a few minutes old.",
    });
  }
  return signals;
}
