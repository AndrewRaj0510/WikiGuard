"use client";

import type { ApiError } from "@/lib/api/client";
import { relativeTime } from "@/lib/format";
import { useNow } from "@/lib/use-now";
import { cn } from "@/lib/utils";

/** What the data on screen is: live, paused, reconnecting, or served from the API's stale cache. */
export function LiveStatus({
  paused,
  error,
  updatedAt,
  serverStale,
}: {
  paused: boolean;
  error: ApiError | undefined;
  updatedAt: number | undefined;
  serverStale?: boolean;
}) {
  const now = useNow();
  const [label, tone] = paused
    ? ["Paused while hidden", "idle"]
    : error
      ? ["Reconnecting", "warn"]
      : serverStale
        ? ["Showing cached data", "warn"]
        : ["Live", "live"];
  const age = updatedAt ? relativeTime(new Date(updatedAt).toISOString(), now) : "";
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
      {age && <span className="tabular">· updated {age}</span>}
    </p>
  );
}
