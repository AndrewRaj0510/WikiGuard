"use client";

import { useState } from "react";
import { Sparkles } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";
import { ApiError } from "@/lib/api/client";
import type { ExplainReply } from "@/lib/api/types";
import { useSeat, useSessionStore } from "@/lib/session/provider";
import { useShortcuts } from "@/lib/shortcuts";

/** Why an answer is "unavailable", in the API's words (services/api/wikiguard_api/explain.py). */
const REASON: Record<string, string> = {
  daily_cap: "Today’s explanation budget is used up. Saved explanations still work.",
  guest_limit: "You’ve used this session’s fresh explanations. Saved ones still work.",
  rejected_output: "The model’s answer failed WikiGuard’s safety checks, so it isn’t shown.",
  hidden: "Wikipedia has hidden this revision, so it isn’t sent to the model.",
  read_only: "WikiGuard is read-only right now.",
  llm_error: "The language model didn’t answer. Try again later.",
};

type State =
  | { status: "idle" }
  | { status: "loading" }
  | { status: "done"; reply: ExplainReply }
  | { status: "failed"; message: string };

function failureText(error: unknown): string {
  if (!(error instanceof ApiError)) return "Something went wrong.";
  if (error.code === "feature_disabled") return "Explain is switched off on this server.";
  if (error.status === 429) {
    const wait = error.retryAfterSeconds;
    return wait ? `Too many requests. Try again in ${wait} s.` : "Too many requests for a moment.";
  }
  if (error.unreachable) return "WikiGuard’s API isn’t answering.";
  return error.detail ?? error.message;
}

export function ExplainPanel({ revId, hasSaved }: { revId: number; hasSaved: boolean }) {
  const store = useSessionStore();
  const seat = useSeat();
  const [state, setState] = useState<State>({ status: "idle" });
  const guestLeft = seat.role === "guest" ? seat.freshExplainsLeft : null;

  async function explain() {
    setState({ status: "loading" });
    try {
      const reply = await store.api.explain(revId);
      setState({ status: "done", reply });
      if (!reply.cached) void store.refresh();
    } catch (error) {
      setState({ status: "failed", message: failureText(error) });
    }
  }

  useShortcuts({ e: () => void explain() }, state.status === "idle" || state.status === "failed");

  return (
    <section aria-labelledby="explain-heading" className="rounded-lg border bg-card p-5">
      <div className="flex items-center justify-between gap-3">
        <h2 id="explain-heading" className="flex items-center gap-2 text-sm font-semibold">
          <Sparkles className="size-4 text-risk-elevated" aria-hidden /> Explain
        </h2>
        {guestLeft !== null && (
          <p className="font-mono text-[11.5px] text-muted-foreground tabular">
            {guestLeft} fresh left
          </p>
        )}
      </div>

      <div aria-live="polite" className="mt-3">
        {state.status === "idle" && (
          <p className="text-[13.5px] leading-relaxed text-muted-foreground">
            A one-line summary of what this edit did, written by a language model from the stored
            diff. {hasSaved ? "A saved one exists, so it’s free." : ""}
          </p>
        )}
        {state.status === "loading" && (
          <div aria-label="Writing an explanation" className="space-y-2">
            <Skeleton className="h-3.5 w-full" />
            <Skeleton className="h-3.5 w-4/5" />
          </div>
        )}
        {state.status === "done" &&
          (state.reply.available ? (
            <figure>
              {/* Model output is untrusted too: plain text only, never HTML or markdown. */}
              <blockquote className="border-l-2 border-risk-elevated/60 pl-3 font-display text-[16.5px] leading-snug">
                {state.reply.explanation}
              </blockquote>
              <figcaption className="mt-2 text-[11.5px] text-muted-foreground">
                AI-written, may be wrong · {state.reply.cached ? "saved answer" : "fresh answer"}
                {state.reply.llm_model ? ` · ${state.reply.llm_model}` : ""}
              </figcaption>
            </figure>
          ) : (
            <p className="text-[13.5px] text-muted-foreground">
              {REASON[state.reply.reason ?? ""] ?? "Explanation unavailable."}
            </p>
          ))}
        {state.status === "failed" && (
          <p role="alert" className="text-[13.5px] text-risk-flagged">
            {state.message}
          </p>
        )}
      </div>

      {state.status !== "done" && (
        <Button
          variant="outline"
          size="sm"
          className="mt-4"
          disabled={state.status === "loading"}
          aria-keyshortcuts="e"
          onClick={() => void explain()}
        >
          <Sparkles aria-hidden />
          {state.status === "loading" ? "Explaining…" : "Explain this edit"}
        </Button>
      )}
    </section>
  );
}
