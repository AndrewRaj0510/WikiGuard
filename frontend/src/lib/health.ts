/**
 * Reading the Health summary's free-form parts. live_metrics, drift_reports, and retrain_runs are
 * written by the monitor and trainer (jobs/monitor, trainer.lifecycle), so each field is checked
 * before use, and anything malformed is skipped instead of breaking the page.
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
  /** Why it ran: the monitor's trigger reasons, or the owner's reason for a manual one. */
  reasons: string[];
  /** Names of the promotion gate's failed checks, when the gate ran and failed. */
  gateFailures: string[];
  /** The version in its shadow period, while status is "shadow". */
  candidateVersion: string | null;
  shadowStartedAt: string | null;
  /** True for the owner's manual override, which skips the gate. */
  forced: boolean;
  /** Section 4's post-promotion check, once the monitor has run it. */
  postCheck: { due: boolean; rollback: boolean; reason: string } | null;
  rolledBack: { at: string | null; reason: string | null; by: string | null } | null;
}

export type RunTone = "ok" | "warn" | "danger" | "outline";

/** What each status the trainer writes means, in words, and how it's shown. */
export function runStatus(status: string): { label: string; tone: RunTone } {
  switch (status) {
    case "promoted":
      return { label: "Promoted", tone: "ok" };
    case "done":
      return { label: "Done", tone: "ok" };
    case "shadow":
      return { label: "In shadow", tone: "warn" };
    case "not_promoted":
      return { label: "Champion kept", tone: "outline" };
    case "rejected":
      return { label: "Rejected", tone: "danger" };
    case "rolled_back":
      return { label: "Rolled back", tone: "danger" };
    case "failed":
      return { label: "Failed", tone: "danger" };
    case "running":
      return { label: "Running", tone: "outline" };
    default:
      return { label: status.replaceAll("_", " "), tone: "outline" };
  }
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

const strings = (v: unknown): string[] =>
  Array.isArray(v) ? v.filter((item): item is string => typeof item === "string" && !!item) : [];

function gateFailuresOf(details: Json): string[] {
  const gate = obj(details.gate);
  if (gate.passed !== false || !Array.isArray(gate.checks)) return [];
  return gate.checks
    .map(obj)
    .filter((check) => check.passed === false)
    .map((check) => str(check.name))
    .filter((name): name is string => name !== null);
}

function postCheckOf(details: Json): RetrainRun["postCheck"] {
  const check = obj(details.post_check);
  const reason = str(check.reason);
  if (typeof check.rollback !== "boolean" || !reason) return null;
  return { due: check.due === true, rollback: check.rollback, reason };
}

export function retrainRunsOf(summary: HealthSummary): RetrainRun[] {
  return summary.retrain_runs
    .filter((row) => typeof row.started_at === "string")
    .map((row) => {
      const details = obj(row.details);
      const reasons = strings(details.reasons);
      const asked = str(details.reason);
      const rolled = str(details.rolled_back_at);
      return {
        id: num(row.id),
        trigger: str(row.trigger) ?? "unknown",
        status: str(row.status) ?? "unknown",
        startedAt: row.started_at as string,
        finishedAt: str(row.finished_at),
        comparison: comparisonOf(row.details),
        reasons: reasons.length ? reasons : row.trigger === "manual" && asked ? [asked] : [],
        gateFailures: gateFailuresOf(details),
        candidateVersion:
          details.candidate_version === undefined || details.candidate_version === null
            ? null
            : String(details.candidate_version),
        shadowStartedAt: str(details.shadow_started_at),
        forced: details.forced === true,
        postCheck: postCheckOf(details),
        rolledBack:
          row.status === "rolled_back" || rolled
            ? { at: rolled, reason: str(details.rollback_reason), by: str(details.rolled_back_by) }
            : null,
      };
    });
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
