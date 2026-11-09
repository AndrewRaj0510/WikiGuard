"use client";

import { useState } from "react";
import { ArrowRight, Clock, Copy, KeyRound, PlugZap, ShieldAlert } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";
import { describeError } from "@/components/state-notice";
import { formatCountdown } from "@/lib/format";
import { useSession, useSessionStore } from "@/lib/session/provider";
import type { SessionState } from "@/lib/session/store";
import { useNow } from "@/lib/use-now";
import { usePoll } from "@/lib/use-poll";

const ENDED_NOTE = {
  expired: "Your guest session reached its time limit, so the seat is free for the next visitor.",
  taken_over: "The owner signed in and took the seat.",
  signed_out: "You signed out.",
  ended: "Your session has ended.",
} as const;

// PLAN section 16: a visitor who finds the app in use sees it recheck every 30 seconds.
const RECHECK_MS = 30_000;
// Free Render sleeps, and the first request wakes it (F-105). Say so once a start is this slow.
const WAKING_AFTER_MS = 3_000;
const READY_POLL_MS = 5_000;

/**
 * Shown in place of the console until this tab holds the seat. The server enforces all of this;
 * these screens only explain it and offer the next step.
 */
export function GateScreen() {
  const state = useSession();
  return (
    <div className="mx-auto flex min-h-[62vh] max-w-lg items-center px-4 py-16">
      <div className="w-full animate-rise rounded-lg border bg-card p-8 shadow-[0_1px_0_0_oklch(1_0_0/0.5)_inset,0_12px_32px_-16px_oklch(0_0_0/0.25)]">
        <GateBody state={state} />
      </div>
    </div>
  );
}

function GateBody({ state }: { state: SessionState }) {
  switch (state.status) {
    case "starting":
    case "seated":
      return <Starting since={state.status === "starting" ? state.since : 0} />;
    case "signed_out":
    case "signing_in":
      return <SignIn state={state} />;
    case "seat_taken":
      return <InUse retryAfterSeconds={state.retryAfterSeconds} />;
    case "confirm_take_over":
      return <ConfirmTakeOver holder={state.holderRole} />;
    case "cooldown":
      return <Cooldown seconds={state.retryAfterSeconds} />;
    case "other_tab":
      return <OtherTab byLock={state.byLock ?? false} />;
    case "lost":
      return <Lost />;
    case "unreachable":
      return <Waking message={describeError(state.error)} />;
  }
}

function Starting({ since }: { since: number }) {
  const now = useNow();
  if (since > 0 && now - since > WAKING_AFTER_MS) {
    return (
      <div aria-live="polite">
        <Eyebrow icon={<Clock aria-hidden className="size-3.5" />}>Waking up</Eyebrow>
        <Heading>Starting WikiGuard’s server</Heading>
        <p className="mt-3 text-[15px] leading-relaxed text-muted-foreground">
          The free server sleeps when nobody’s using it, and the first visit wakes it. That can
          take up to a minute. This page carries on by itself.
        </p>
        <WakingBar />
      </div>
    );
  }
  return (
    <div aria-busy="true" aria-label="Connecting" className="space-y-4">
      <Skeleton className="h-7 w-2/3" />
      <Skeleton className="h-4 w-full" />
      <Skeleton className="h-9 w-40" />
    </div>
  );
}

function WakingBar() {
  return (
    <div className="mt-6 h-1 overflow-hidden rounded-full bg-muted" aria-hidden>
      <div className="h-full w-1/3 animate-shimmer rounded-full bg-[linear-gradient(90deg,transparent,var(--risk-elevated),transparent)] bg-[length:200%_100%]" />
    </div>
  );
}

function SignIn({ state }: { state: Extract<SessionState, { status: "signed_out" | "signing_in" }> }) {
  const store = useSessionStore();
  const signingIn = state.status === "signing_in";
  const [showOwner, setShowOwner] = useState(
    state.status === "signed_out" && Boolean(state.loginError),
  );
  const [password, setPassword] = useState("");

  function submit(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!password) return;
    const typed = password;
    // Don't keep it on screen or in state any longer than the request needs.
    setPassword("");
    void store.signInUser(typed);
  }

  return (
    <>
      <Eyebrow>Read-only demo</Eyebrow>
      <Heading>Step into the review desk</Heading>
      {state.status === "signed_out" && state.reason && (
        <p className="mt-3 rounded-md bg-muted px-3 py-2 text-sm text-muted-foreground">
          {ENDED_NOTE[state.reason]}
        </p>
      )}
      <p className="mt-3 text-[15px] leading-relaxed text-muted-foreground">
        A guest can read the live feed, the review queue, every scored edit, and the model pages,
        and ask for five fresh explanations. Sessions last up to 15 minutes, or 5 idle, because
        one person uses WikiGuard at a time.
      </p>
      <Button
        size="lg"
        className="mt-6"
        disabled={signingIn}
        onClick={() => void store.signInGuest()}
      >
        {signingIn && state.as === "guest" ? "Signing in…" : "Continue as guest"}
        <ArrowRight aria-hidden />
      </Button>

      <div className="mt-6 border-t pt-4">
        {!showOwner ? (
          <button
            type="button"
            className="inline-flex items-center gap-1.5 rounded-sm text-[13px] font-medium text-muted-foreground hover:text-foreground"
            onClick={() => setShowOwner(true)}
          >
            <KeyRound className="size-3.5" aria-hidden /> Owner sign-in
          </button>
        ) : (
          <form onSubmit={submit} className="space-y-2" aria-label="Owner sign-in">
            <label htmlFor="owner-password" className="block text-[13px] font-medium">
              Owner password
            </label>
            <div className="flex gap-2">
              <input
                id="owner-password"
                type="password"
                autoComplete="current-password"
                autoFocus
                value={password}
                onChange={(event) => setPassword(event.target.value)}
                maxLength={256}
                aria-invalid={state.status === "signed_out" && Boolean(state.loginError)}
                aria-describedby={state.status === "signed_out" && state.loginError ? "login-error" : undefined}
                className="h-9 min-w-0 flex-1 rounded-md border bg-background px-3 text-sm outline-none focus-visible:outline-2 focus-visible:outline-ring"
              />
              <Button type="submit" variant="outline" disabled={signingIn || !password}>
                {signingIn && state.as === "user" ? "Signing in…" : "Sign in"}
              </Button>
            </div>
            {state.status === "signed_out" && state.loginError && (
              <p id="login-error" role="alert" className="text-[13px] text-risk-flagged">
                {state.loginError}
              </p>
            )}
          </form>
        )}
      </div>

      <p className="mt-5 text-xs text-muted-foreground">
        Edit text comes straight from Wikipedia and may include vandalism.
      </p>
    </>
  );
}

function InUse({ retryAfterSeconds }: { retryAfterSeconds: number | null }) {
  const store = useSessionStore();
  const now = useNow();
  // The public gate status, every 30 s, so the page notices the seat coming free by itself.
  const gate = usePoll("gate-status", (signal) => store.api.gateStatus(signal), {
    intervalMs: RECHECK_MS,
  });
  const [shownAt] = useState(() => Date.now());
  const free = gate.data?.state === "free";
  const left =
    gate.data?.retry_after_seconds ??
    (retryAfterSeconds !== null && now > 0
      ? Math.max(0, retryAfterSeconds - (now - shownAt) / 1000)
      : null);

  if (free) {
    return (
      <div aria-live="polite">
        <Eyebrow>Free now</Eyebrow>
        <Heading>The seat just came free</Heading>
        <p className="mt-3 text-[15px] text-muted-foreground">Go ahead before someone else does.</p>
        <Button size="lg" className="mt-6" onClick={() => void store.signInGuest()}>
          Continue as guest <ArrowRight aria-hidden />
        </Button>
        <button
          type="button"
          className="mt-4 block text-[13px] font-medium text-muted-foreground hover:text-foreground"
          onClick={() => store.showSignIn()}
        >
          Owner sign-in
        </button>
      </div>
    );
  }
  return (
    <div aria-live="polite">
      <Eyebrow icon={<Clock aria-hidden className="size-3.5" />}>In use</Eyebrow>
      <Heading>Someone else is at the desk</Heading>
      <p className="mt-3 text-[15px] leading-relaxed text-muted-foreground">
        Only one person can use WikiGuard at a time.{" "}
        {left !== null && left > 0
          ? `Try again in about ${Math.max(1, Math.ceil(left / 60))} minute${left > 60 ? "s" : ""}.`
          : "Try again shortly."}{" "}
        This page checks every 30 seconds.
      </p>
      {left !== null && left > 0 && (
        <p className="mt-4 font-mono text-2xl tabular">{formatCountdown(left)}</p>
      )}
      <div className="mt-6 flex flex-wrap gap-3">
        <Button variant="outline" onClick={gate.refresh}>
          Check now
        </Button>
        <Button variant="ghost" onClick={() => store.showSignIn()}>
          <KeyRound aria-hidden /> I’m the owner
        </Button>
      </div>
    </div>
  );
}

function ConfirmTakeOver({ holder }: { holder: "guest" | "user" | null }) {
  const store = useSessionStore();
  const who = holder === "user" ? "your own session on another device" : "a guest";
  return (
    <>
      <Eyebrow icon={<ShieldAlert aria-hidden className="size-3.5" />}>Confirm</Eyebrow>
      <Heading>WikiGuard is in use</Heading>
      <p className="mt-3 text-[15px] leading-relaxed text-muted-foreground">
        It’s held by {who}. Taking the seat ends that session at once, and it’s recorded in the
        audit log.
      </p>
      <div className="mt-6 flex flex-wrap gap-3">
        <Button onClick={() => void store.confirmTakeOver()}>
          Take the seat <ArrowRight aria-hidden />
        </Button>
        <Button variant="ghost" onClick={() => store.cancelTakeOver()}>
          Cancel
        </Button>
      </div>
    </>
  );
}

function Cooldown({ seconds }: { seconds: number }) {
  const store = useSessionStore();
  const now = useNow();
  const [shownAt] = useState(() => Date.now());
  const left = now > 0 ? Math.max(0, seconds - (now - shownAt) / 1000) : seconds;
  return (
    <>
      <Eyebrow icon={<Clock aria-hidden className="size-3.5" />}>One moment</Eyebrow>
      <Heading>Please wait a little</Heading>
      <p className="mt-3 text-[15px] text-muted-foreground">
        A guest session just ended from this network, so the next one waits a minute. That keeps
        one visitor from holding the seat forever.
      </p>
      <p className="mt-4 font-mono text-2xl tabular" aria-live="off">
        {formatCountdown(left)}
      </p>
      <Button
        variant="outline"
        className="mt-6"
        disabled={left > 0}
        onClick={() => void store.signInGuest()}
      >
        Continue as guest
      </Button>
    </>
  );
}

function OtherTab({ byLock }: { byLock: boolean }) {
  const store = useSessionStore();
  return (
    <>
      <Eyebrow icon={<Copy aria-hidden className="size-3.5" />}>Another tab</Eyebrow>
      <Heading>WikiGuard is open in another tab</Heading>
      <p className="mt-3 text-[15px] text-muted-foreground">
        {byLock
          ? "This browser already has WikiGuard open. Use that tab, or continue here and it stops there."
          : "Your session can be active in one tab at a time. Continuing here closes it there."}
      </p>
      <Button className="mt-6" onClick={() => void store.useHere()}>
        Use it here <ArrowRight aria-hidden />
      </Button>
    </>
  );
}

function Lost() {
  const store = useSessionStore();
  return (
    <>
      <Eyebrow icon={<Clock aria-hidden className="size-3.5" />}>Seat lost</Eyebrow>
      <Heading>This tab lost its seat</Heading>
      <p className="mt-3 text-[15px] text-muted-foreground">
        Its hold on WikiGuard ran out, which can happen after a laptop sleeps. Try to take it back.
      </p>
      <Button className="mt-6" onClick={() => void store.start()}>
        Try again <ArrowRight aria-hidden />
      </Button>
    </>
  );
}

/** Can't reach the API. Most often it's asleep, so watch /ready and carry on when it answers. */
function Waking({ message }: { message: string }) {
  const store = useSessionStore();
  const ready = usePoll(
    "ready",
    async (signal) => {
      const answer = await store.api.ready(signal);
      if (answer.ready) void store.start();
      return answer;
    },
    { intervalMs: READY_POLL_MS },
  );
  return (
    <div aria-live="polite">
      <Eyebrow icon={<PlugZap aria-hidden className="size-3.5" />}>Can’t connect yet</Eyebrow>
      <Heading>WikiGuard isn’t answering</Heading>
      <p className="mt-3 text-[15px] text-muted-foreground">{message}</p>
      <p className="mt-2 text-[13px] text-muted-foreground">
        {ready.error ? "Still trying every few seconds." : "It’s answering again. Connecting…"}
      </p>
      <WakingBar />
      <Button variant="outline" className="mt-6" onClick={() => void store.start()}>
        Try now
      </Button>
    </div>
  );
}

function Eyebrow({ children, icon }: { children: React.ReactNode; icon?: React.ReactNode }) {
  return (
    <p className="flex items-center gap-1.5 text-[11px] font-medium tracking-[0.12em] text-risk-flagged uppercase">
      {icon}
      {children}
    </p>
  );
}

function Heading({ children }: { children: React.ReactNode }) {
  return (
    <h1 className="mt-2 font-display text-[28px] leading-tight font-medium tracking-[-0.01em]">
      {children}
    </h1>
  );
}
