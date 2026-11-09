"use client";

import type { ModelEntry } from "./api/types";
import { chainRank, statsOf, type ModelStats } from "./models";
import { useSessionStore } from "./session/provider";
import { usePoll } from "./use-poll";

// The API caches this for 5 minutes, and it changes only when a model is promoted.
const MODELS_POLL_MS = 300_000;

export interface ModelRow {
  entry: ModelEntry;
  stats: ModelStats;
}

/** The registered models, in serving order: champion, then fallbacks, then the rest. */
export function useModels() {
  const store = useSessionStore();
  const poll = usePoll("models", (signal) => store.api.models(signal), {
    intervalMs: MODELS_POLL_MS,
  });
  const rows: ModelRow[] = (poll.data?.items ?? [])
    .map((entry) => ({ entry, stats: statsOf(entry) }))
    .sort((a, b) => chainRank(a.entry.alias) - chainRank(b.entry.alias));
  return { ...poll, rows };
}

/** Precision at each whole percent of recall, as the PR chart needs it, for models with a curve. */
export function curveSeries(rows: ModelRow[]) {
  return rows
    .filter((row) => row.stats.curve)
    .map((row) => ({
      name: row.entry.model_name,
      precision: row.stats.curve!.points.map(([, precision]) => precision),
    }));
}
