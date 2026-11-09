/**
 * Reading a model's stored metrics. They arrive as free-form JSON (`metrics: dict[str, Any]` in
 * the API), written by scripts/sync_model_versions.py, so every field is checked before use: a
 * missing or malformed number shows as "not measured" instead of breaking the page.
 */
import type { ModelEntry } from "./api/types";

export type Interval = [number, number];

export interface SliceStats {
  rows: number;
  positives: number | null;
  prAuc: number | null;
  falsePositiveRate: number | null;
}

export interface PrCurve {
  /** [recall, precision] pairs, recall rising. */
  points: Interval[];
  rows: number | null;
  rowsScored: number | null;
  prevalence: number | null;
}

export interface ModelStats {
  prAuc: number | null;
  prAucInterval: Interval | null;
  recallAt90: number | null;
  recallAt90Interval: Interval | null;
  precisionAt50: number | null;
  rocAuc: number | null;
  eceAfter: number | null;
  eceBefore: number | null;
  servedBytes: number | null;
  laptopP50Ms: number | null;
  laptopP95Ms: number | null;
  warnings: string[];
  thresholdRule: string | null;
  atThreshold: { precision: number | null; recall: number | null; f1: number | null };
  trainRows: number | null;
  testRows: number | null;
  testPositives: number | null;
  parameters: number | null;
  device: string | null;
  fitMinutes: number | null;
  parityGap: number | null;
  goldenPrAuc: number | null;
  falsePositiveGap: number | null;
  snapshot: string | null;
  slices: Record<string, SliceStats>;
  curve: PrCurve | null;
}

type Json = Record<string, unknown>;

function num(value: unknown): number | null {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

function str(value: unknown): string | null {
  return typeof value === "string" && value.length > 0 ? value : null;
}

function obj(value: unknown): Json {
  return value !== null && typeof value === "object" && !Array.isArray(value)
    ? (value as Json)
    : {};
}

function interval(value: unknown): Interval | null {
  if (!Array.isArray(value) || value.length !== 2) return null;
  const [low, high] = [num(value[0]), num(value[1])];
  return low !== null && high !== null && low <= high ? [low, high] : null;
}

function curveOf(value: unknown): PrCurve | null {
  const raw = obj(value);
  if (!Array.isArray(raw.points)) return null;
  const points = raw.points
    .filter((p): p is [number, number] => Array.isArray(p) && num(p[0]) !== null && num(p[1]) !== null)
    .map(([recall, precision]) => [recall, precision] as Interval);
  if (points.length < 2) return null;
  return {
    points,
    rows: num(raw.rows),
    rowsScored: num(raw.rows_scored),
    prevalence: num(raw.prevalence),
  };
}

export function statsOf(entry: ModelEntry): ModelStats {
  const metrics = obj(entry.metrics);
  const card = obj(metrics.card);
  const at = obj(card.at_threshold);
  const slices: Record<string, SliceStats> = {};
  for (const [name, raw] of Object.entries(obj(card.slices))) {
    const slice = obj(raw);
    const rows = num(slice.rows);
    if (rows === null) continue;
    slices[name] = {
      rows,
      positives: num(slice.positives),
      prAuc: num(slice.pr_auc),
      falsePositiveRate: num(slice.false_positive_rate),
    };
  }
  return {
    prAuc: num(metrics.pr_auc),
    prAucInterval: interval(metrics.pr_auc_interval),
    recallAt90: num(metrics.recall_at_90_precision),
    recallAt90Interval: interval(metrics.recall_at_90_precision_interval),
    precisionAt50: num(metrics.precision_at_50),
    rocAuc: num(metrics.roc_auc),
    eceAfter: num(metrics.ece_after),
    eceBefore: num(card.ece_before),
    servedBytes: num(metrics.served_bytes),
    laptopP50Ms: num(metrics.laptop_p50_ms),
    laptopP95Ms: num(metrics.laptop_p95_ms),
    warnings: Array.isArray(metrics.training_warnings)
      ? metrics.training_warnings.filter((w): w is string => typeof w === "string")
      : [],
    thresholdRule: str(card.threshold_rule),
    atThreshold: { precision: num(at.precision), recall: num(at.recall), f1: num(at.f1) },
    trainRows: num(card.train_rows),
    testRows: num(card.rows),
    testPositives: num(card.positives),
    parameters: num(obj(card.parameters).total),
    device: str(card.device_name),
    fitMinutes: num(card.fit_minutes),
    parityGap: num(card.parity_gap),
    goldenPrAuc: num(card.golden_pr_auc),
    falsePositiveGap: num(card.false_positive_gap_anonymous_minus_registered),
    snapshot: str(card.snapshot),
    slices,
    curve: curveOf(card.pr_curve),
  };
}

export const MODEL_LABEL: Record<string, string> = {
  roberta: "RoBERTa",
  distilbert: "DistilBERT",
  bert: "BERT",
  bilstm: "BiLSTM",
  mlp: "MLP",
};

export function modelLabel(name: string): string {
  return MODEL_LABEL[name] ?? name;
}

/** The CSS variable holding a model's chart color. Unknown names get a neutral ink. */
export function seriesColor(name: string): string {
  return name in MODEL_LABEL ? `var(--series-${name})` : "var(--muted-foreground)";
}

/** "champion" and "fallback-N" are registry aliases (CLAUDE.md); show them as words. */
export function aliasLabel(alias: string | null): string {
  if (!alias) return "Registered";
  if (alias === "champion") return "Champion";
  const match = /^fallback-(\d+)$/.exec(alias);
  return match ? `Fallback ${match[1]}` : alias;
}

/** The serving order: champion, then fallbacks by number, then anything without an alias. */
export function chainRank(alias: string | null): number {
  if (alias === "champion") return 0;
  const match = alias ? /^fallback-(\d+)$/.exec(alias) : null;
  return match ? Number(match[1]) : 1000;
}

export const SLICE_LABEL: Record<string, string> = {
  anonymous: "Logged out",
  registered: "Registered",
  new_editor: "New editors (up to 10 edits)",
  experienced_editor: "Experienced editors (more than 10)",
  small_edit: "Small edits (100 bytes or less)",
  large_edit: "Large edits (over 100 bytes)",
};

export function sliceLabel(name: string): string {
  return SLICE_LABEL[name] ?? name.replaceAll("_", " ");
}
