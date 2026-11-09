"use client";

import { useState } from "react";
import { Check, Lock } from "lucide-react";
import { Button } from "@/components/ui/button";
import { ApiError } from "@/lib/api/client";
import type { Decision } from "@/lib/api/types";
import { useSeat, useSessionStore } from "@/lib/session/provider";
import { useShortcuts } from "@/lib/shortcuts";
import { cn } from "@/lib/utils";

export const DECISIONS: { value: Decision; label: string; key: string }[] = [
  { value: "vandalism", label: "Vandalism", key: "v" },
  { value: "good_faith_mistake", label: "Good-faith mistake", key: "g" },
  { value: "ok", label: "OK", key: "o" },
  { value: "skip", label: "Skip", key: "s" },
];

const LABEL: Record<Decision, string> = Object.fromEntries(
  DECISIONS.map((d) => [d.value, d.label]),
) as Record<Decision, string>;

function failureText(error: unknown): string {
  if (!(error instanceof ApiError)) return "Something went wrong.";
  if (error.code === "read_only") return "WikiGuard is read-only right now, so reviews can’t be saved.";
  if (error.code === "forbidden") return "Only the owner’s account can review edits.";
  if (error.unreachable) return "WikiGuard’s API isn’t answering.";
  return error.detail ?? error.message;
}

/**
 * The owner records a decision here. A guest sees the same controls switched off, because the
 * server refuses a guest anyway (PLAN section 16): hiding a button isn't protection.
 * WikiGuard never reverts anything; this only records what a person decided.
 */
export function ReviewControls({
  revId,
  current,
  onSaved,
}: {
  revId: number;
  current: Decision | null;
  onSaved: (decision: Decision) => void;
}) {
  const store = useSessionStore();
  const seat = useSeat();
  const isOwner = seat.role === "user";
  const [state, setState] = useState<
    { status: "idle" } | { status: "saving"; decision: Decision } | { status: "failed"; message: string }
  >({ status: "idle" });
  const [note, setNote] = useState("");

  async function save(decision: Decision) {
    if (!isOwner || state.status === "saving") return;
    setState({ status: "saving", decision });
    try {
      await store.api.review(revId, decision, note.trim() || undefined);
      setState({ status: "idle" });
      setNote("");
      onSaved(decision);
    } catch (error) {
      setState({ status: "failed", message: failureText(error) });
    }
  }

  useShortcuts(
    Object.fromEntries(DECISIONS.map((d) => [d.key, () => void save(d.value)])),
    isOwner,
  );

  return (
    <section aria-labelledby="review-heading" className="rounded-lg border bg-card p-5">
      <div className="flex items-center justify-between gap-2">
        <h2 id="review-heading" className="text-sm font-semibold">
          Your decision
        </h2>
        {current && (
          <p className="flex items-center gap-1 text-[12.5px] text-diff-added" role="status">
            <Check className="size-3.5" aria-hidden />
            <span className="text-foreground">{LABEL[current]}</span>
          </p>
        )}
      </div>
      {!isOwner && (
        <p className="mt-2 flex items-start gap-1.5 text-[12.5px] text-muted-foreground">
          <Lock className="mt-0.5 size-3.5 shrink-0" aria-hidden />
          Reviewing needs the owner’s account. Guests can read everything.
        </p>
      )}
      <div className="mt-3 grid gap-1.5">
        {DECISIONS.map((decision) => (
          <Button
            key={decision.value}
            variant={current === decision.value ? "default" : "outline"}
            size="sm"
            disabled={!isOwner || state.status === "saving"}
            aria-pressed={current === decision.value}
            aria-keyshortcuts={isOwner ? decision.key : undefined}
            onClick={() => void save(decision.value)}
            className={cn("justify-between", decision.value === "vandalism" && current !== "vandalism" && "hover:border-risk-flagged/50")}
          >
            {state.status === "saving" && state.decision === decision.value ? "Saving…" : decision.label}
            {isOwner && (
              <kbd className="rounded border px-1 font-mono text-[10.5px] text-muted-foreground">
                {decision.key}
              </kbd>
            )}
          </Button>
        ))}
      </div>
      {isOwner && (
        <>
          <label htmlFor="review-note" className="mt-3 block text-[12.5px] text-muted-foreground">
            Note (optional)
          </label>
          <input
            id="review-note"
            value={note}
            maxLength={500}
            onChange={(event) => setNote(event.target.value)}
            className="mt-1 h-8 w-full rounded-md border bg-background px-2 text-[13px]"
          />
        </>
      )}
      {state.status === "failed" && (
        <p role="alert" className="mt-2 text-[12.5px] text-risk-flagged">
          {state.message}
        </p>
      )}
    </section>
  );
}
