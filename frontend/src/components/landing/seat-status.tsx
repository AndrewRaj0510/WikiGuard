"use client";

import { createApiClient } from "@/lib/api/client";
import { usePoll } from "@/lib/use-poll";
import { cn } from "@/lib/utils";

// The public gate status needs no session, so this client sends no tab ID and no token.
const publicApi = createApiClient({ credentials: () => ({ tabId: null, csrfToken: null }) });
const RECHECK_MS = 30_000;

/**
 * Whether the one seat is free, before a visitor clicks in. It says nothing about who's using it
 * (PLAN section 16), and it stays quiet if the API can't be reached.
 */
export function SeatStatus() {
  const gate = usePoll("landing-gate", (signal) => publicApi.gateStatus(signal), {
    intervalMs: RECHECK_MS,
  });
  if (!gate.data) return <p className="h-5" aria-hidden />;
  const inUse = gate.data.enabled && gate.data.state === "in_use";
  const minutes = gate.data.retry_after_seconds
    ? Math.max(1, Math.ceil(gate.data.retry_after_seconds / 60))
    : null;
  return (
    <p className="flex h-5 items-center gap-2 text-[13px] text-muted-foreground" aria-live="polite">
      <span
        aria-hidden
        className={cn(
          "size-2 rounded-full",
          inUse ? "bg-risk-elevated" : "animate-pulse-dot bg-diff-added text-diff-added",
        )}
      />
      {inUse ? (
        <span>
          <span className="font-medium text-foreground">In use right now.</span>{" "}
          {minutes ? `Free in about ${minutes} minute${minutes === 1 ? "" : "s"}.` : "Try again shortly."}
        </span>
      ) : (
        <span>
          <span className="font-medium text-foreground">Free now.</span> You can start straight away.
        </span>
      )}
    </p>
  );
}
