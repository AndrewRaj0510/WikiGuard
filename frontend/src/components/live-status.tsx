"use client";

import type { ApiError } from "@/lib/api/client";
import { relativeTime } from "@/lib/format";
import { STALE_SCORING_MINUTES } from "@/lib/health";
import { useNow } from "@/lib/use-now";
import { cn } from "@/lib/utils";

/**
 * What the data on screen is: live, paused, reconnecting, served from the API's stale cache,
 * delayed, or the demo's fixed snapshot, which is never called live (DEC-091).
 *
 * "Live" alone only says the API answered. With `newestAt`, the newest edit's time, a stream that
 * has stopped shows as "Delayed" after the same 10 minutes as the Stale banner, instead of a live
 * dot beside edits from long ago. Failure injection found the two disagreeing (F-158).
 */
export function LiveStatus({
  paused,
  error,
  updatedAt,
  serverStale,
  snapshot = false,
  newestAt,
}: {
  paused: boolean;
  error: ApiError | undefined;
  updatedAt: number | undefined;
  serverStale?: boolean;
  snapshot?: boolean;
  newestAt?: string;
}) {
  const now = useNow();
  const newest = newestAt ? Date.parse(newestAt) : Number.NaN;
  const delayed =
    !snapshot &&
    now > 0 &&
    Number.isFinite(newest) &&
    now - newest >= STALE_SCORING_MINUTES * 60_000;
  const [label, tone] = paused
    ? ["Paused while hidden", "idle"]
    : error
      ? ["Reconnecting", "warn"]
      : serverStale
        ? ["Showing cached data", "warn"]
        : snapshot
          ? ["Snapshot, not live", "idle"]
          : delayed
            ? ["Delayed", "warn"]
            : ["Live", "live"];
  const age = updatedAt ? relativeTime(new Date(updatedAt).toISOString(), now) : "";
  const detail =
    delayed && newestAt ? `· newest edit ${relativeTime(newestAt, now)}` : age && `· updated ${age}`;
  return (
    <p
      className="inline-flex items-center gap-2 text-[12.5px] text-muted-foreground"
      aria-live="polite"
    >
      <span
        aria-hidden
        className={cn(
          "size-2 rounded-full",
          tone === "live" && "animate-pulse-dot bg-diff-added text-diff-added",
          tone === "warn" && "bg-risk-elevated",
          tone === "idle" && "bg-muted-foreground/50",
        )}
      />
      <span className="font-medium text-foreground">{label}</span>
      {detail && <span className="tabular">{detail}</span>}
    </p>
  );
}
