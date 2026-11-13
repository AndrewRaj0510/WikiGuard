"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { CircleAlert, Clock, Info } from "lucide-react";
import { formatCountdown } from "@/lib/format";
import { HEALTH_POLL_MS, healthSignals, snapshotNotice } from "@/lib/health";
import { useSeat, useSessionStore } from "@/lib/session/provider";
import { useNow } from "@/lib/use-now";
import { usePoll } from "@/lib/use-poll";
import { cn } from "@/lib/utils";

const GUEST_WARNING_SECONDS = 60;

/**
 * One line under the header when something's wrong: a degraded model chain, scoring that's
 * stopped, cached data, or a guest session about to end. Nothing shows when all is well, except
 * on the demo's snapshot data, which is always labelled as such, on every page (DEC-091).
 * The same health summary the Health page reads, every 60 s, which the API serves from cache.
 */
export function StatusBanner() {
  const store = useSessionStore();
  const seat = useSeat();
  const now = useNow();
  const health = usePoll("health-banner", (signal) => store.api.health(signal), {
    intervalMs: HEALTH_POLL_MS,
  });
  // The Health page lists these itself, so the banner doesn't repeat them there.
  const onHealthPage = usePathname() === "/health";
  const signals = health.data && now && !onHealthPage ? healthSignals(health.data, now) : [];
  const worst = signals[0];
  const secondsLeft =
    seat.guestEndsAt !== null && now > 0 ? Math.max(0, (seat.guestEndsAt - now) / 1000) : null;
  const ending = secondsLeft !== null && secondsLeft <= GUEST_WARNING_SECONDS;

  const notice = health.data ? snapshotNotice(health.data) : null;

  if (!worst && !ending && !notice) return null;
  return (
    <div className="border-b" role="status" aria-live="polite">
      {notice && (
        <p className="flex items-center justify-center gap-2 bg-muted px-4 py-1.5 text-center text-[13px]">
          <Info className="size-3.5 shrink-0" aria-hidden />
          <span>
            <span className="font-semibold">Demo data:</span> {notice}
          </span>
        </p>
      )}
      {ending && (
        <p className="flex items-center justify-center gap-2 bg-risk-flagged-soft px-4 py-1.5 text-[13px]">
          <Clock className="size-3.5" aria-hidden />
          Your guest session ends in{" "}
          <span className="font-mono tabular">{formatCountdown(secondsLeft)}</span>. Any click or
          key keeps an idle session going, up to 15 minutes in all.
        </p>
      )}
      {worst && (
        <p
          className={cn(
            "flex items-center justify-center gap-2 px-4 py-1.5 text-[13px]",
            worst.tone === "danger" ? "bg-risk-flagged-soft" : "bg-risk-elevated-soft",
          )}
        >
          <CircleAlert className="size-3.5 shrink-0" aria-hidden />
          <span>
            <span className="font-semibold">{worst.title}:</span> {worst.detail}{" "}
            <Link href="/health" className="underline underline-offset-2">
              Details
            </Link>
          </span>
        </p>
      )}
    </div>
  );
}
