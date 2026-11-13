"use client";

import { useState } from "react";
import { Activity, CheckCircle2, CircleAlert, CircleDot, Lock, XCircle } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Chip } from "@/components/ui/chip";
import { Skeleton } from "@/components/ui/skeleton";
import { LiveMetricsChart } from "@/components/charts/lazy";
import { LiveStatus } from "@/components/live-status";
import { EmptyState, ErrorState, describeError } from "@/components/state-notice";
import { PageHeader } from "@/components/views/page-header";
import { NotMeasured } from "@/components/models/parts";
import { ApiError } from "@/lib/api/client";
import type { AdminRequestInfo, HealthSummary } from "@/lib/api/types";
import { absoluteTime, relativeTime } from "@/lib/format";
import {
  HEALTH_POLL_MS,
  driftReportsOf,
  healthSignals,
  liveMetricsOf,
  retrainRunsOf,
  runStatus,
  type RetrainRun,
} from "@/lib/health";
import { aliasLabel, modelLabel } from "@/lib/models";
import { useSeat, useSessionStore } from "@/lib/session/provider";
import { useModels } from "@/lib/use-models";
import { useNow } from "@/lib/use-now";
import { usePoll } from "@/lib/use-poll";
import { cn } from "@/lib/utils";


export function HealthView() {
  const store = useSessionStore();
  const health = usePoll("health", (signal) => store.api.health(signal), {
    intervalMs: HEALTH_POLL_MS,
  });
  const summary = health.data;
  return (
    <div className="mx-auto max-w-6xl px-4 py-8">
      <PageHeader
        eyebrow="Operations"
        title="Health"
        lede={
          summary?.data_mode === "snapshot"
            ? "How the system did on the demo’s dataset: scoring, which models are answering, accuracy measured from community reverts, drift, and every retrain. These charts come from the API; Grafana stays private."
            : "How the live system is doing: scoring volume, which models are answering, accuracy measured from community reverts, drift, and every retrain. These charts come from the API; Grafana stays private."
        }
        aside={
          <LiveStatus
            paused={health.paused}
            error={health.error}
            updatedAt={health.updatedAt}
            serverStale={summary?.stale}
            snapshot={summary?.data_mode === "snapshot"}
          />
        }
      />
      {health.status === "loading" && <Skeleton className="mt-8 h-80 w-full" />}
      {health.status === "error" && health.error && (
        <div className="mt-8 rounded-lg border bg-card">
          <ErrorState error={health.error} onRetry={health.refresh} />
        </div>
      )}
      {summary && (
        <div className="mt-8 space-y-8">
          <Signals summary={summary} />
          <Tiles summary={summary} />
          <div className="grid gap-6 lg:grid-cols-[minmax(0,1fr)_22rem]">
            <LiveAccuracy summary={summary} />
            <Chain summary={summary} />
          </div>
          <div className="grid gap-6 lg:grid-cols-[minmax(0,1fr)_22rem]">
            <RetrainHistory runs={retrainRunsOf(summary)} />
            <Drift summary={summary} />
          </div>
          <AdminPanel />
        </div>
      )}
    </div>
  );
}

function Signals({ summary }: { summary: HealthSummary }) {
  const now = useNow();
  const signals = now ? healthSignals(summary, now) : [];
  if (!signals.length) {
    return (
      <p className="flex items-center gap-2 text-[13.5px] text-muted-foreground">
        <CheckCircle2 className="size-4 text-diff-added" aria-hidden />
        <span>
          <span className="font-medium text-foreground">Healthy.</span> Nothing needs attention
          right now.
        </span>
      </p>
    );
  }
  return (
    <ul className="space-y-2">
      {signals.map((signal) => (
        <li
          key={signal.detail}
          className={cn(
            "flex items-start gap-2 rounded-md border px-3 py-2 text-[13.5px]",
            signal.tone === "danger"
              ? "border-risk-flagged/40 bg-risk-flagged-soft"
              : "border-risk-elevated/40 bg-risk-elevated-soft",
          )}
        >
          <CircleAlert className="mt-0.5 size-4 shrink-0" aria-hidden />
          <span>
            <span className="font-semibold">{signal.title}:</span> {signal.detail}
          </span>
        </li>
      ))}
    </ul>
  );
}

function Tiles({ summary }: { summary: HealthSummary }) {
  const now = useNow();
  const snapshot = summary.data_mode === "snapshot";
  const tiles = [
    {
      label: snapshot ? "Edits scored in the dataset" : "Scored in the last 24 h",
      value: summary.scored_last_24h.toLocaleString("en-US"),
    },
    { label: "Scored by a fallback", value: rate(summary.fallback_rate_24h) },
    { label: "Scored by a faster model under load", value: rate(summary.load_shed_rate_24h) },
    {
      label: "Last edit scored",
      value: summary.last_scored_at ? relativeTime(summary.last_scored_at, now) || "—" : "—",
      title: summary.last_scored_at ? absoluteTime(summary.last_scored_at) : undefined,
    },
    { label: "Waiting to be scored", value: summary.unscored_edits.toLocaleString("en-US") },
  ];
  return (
    <section aria-label={snapshot ? "Scoring of the dataset" : "Scoring in the last day"} className="grid gap-px overflow-hidden rounded-lg border bg-border sm:grid-cols-5">
      {tiles.map((tile) => (
        <div key={tile.label} className="bg-card px-5 py-4" title={tile.title}>
          <p className="font-mono text-[26px] leading-none font-medium tracking-tight tabular">
            {tile.value}
          </p>
          <p className="mt-2 text-[12.5px] leading-snug text-muted-foreground">{tile.label}</p>
        </div>
      ))}
    </section>
  );
}

function rate(value: number | null): string {
  return value === null ? "—" : `${(value * 100).toFixed(1)}%`;
}

function LiveAccuracy({ summary }: { summary: HealthSummary }) {
  const models = useModels();
  const champion = models.rows.find((row) => row.entry.alias === "champion");
  const points = liveMetricsOf(summary)
    .filter((row) => row.prAuc !== null)
    .map((row) => ({ day: row.day, prAuc: row.prAuc as number }));
  return (
    <Panel title="Live PR-AUC" icon={<Activity className="size-4" aria-hidden />}>
      <p className="mb-3 text-[13px] text-muted-foreground">
        Measured each day from edits whose 48-hour revert label has arrived, against the
        champion’s offline test result.
      </p>
      {points.length ? (
        <>
          <LiveMetricsChart points={points} offline={champion?.stats.prAuc ?? null} />
          <details className="mt-2 text-[13px]">
            <summary className="cursor-pointer text-muted-foreground hover:text-foreground">
              View as a table
            </summary>
            <table className="mt-2 w-full max-w-sm text-left">
              <thead className="text-[11px] tracking-[0.08em] text-muted-foreground uppercase">
                <tr>
                  <th scope="col" className="py-1 font-medium">Day</th>
                  <th scope="col" className="py-1 text-right font-medium">PR-AUC</th>
                </tr>
              </thead>
              <tbody className="divide-y">
                {points.map((point) => (
                  <tr key={point.day}>
                    <td className="py-1 font-mono">{point.day}</td>
                    <td className="py-1 text-right font-mono tabular">{point.prAuc.toFixed(3)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </details>
        </>
      ) : (
        <EmptyState title="No live measurement yet">
          The daily monitor writes one once labels arrive, 48 hours after each edit.
        </EmptyState>
      )}
    </Panel>
  );
}

const BREAKER = {
  closed: { label: "Answering", Icon: CheckCircle2, tone: "text-diff-added" },
  half_open: { label: "Testing again", Icon: CircleDot, tone: "text-risk-elevated" },
  open: { label: "Switched off", Icon: XCircle, tone: "text-risk-flagged" },
} as const;

function Chain({ summary }: { summary: HealthSummary }) {
  return (
    <Panel title="Serving chain">
      {!summary.models_loaded ? (
        <p className="text-[13px] text-muted-foreground">
          This API process hasn’t loaded its models, so live predictions wait. Stored scores come
          from the scorer, so the queue isn’t affected.
        </p>
      ) : (
        <ol className="space-y-2">
          {summary.chain.map((item, index) => {
            const breaker = BREAKER[item.breaker as keyof typeof BREAKER] ?? {
              label: item.breaker,
              Icon: CircleDot,
              tone: "text-muted-foreground",
            };
            return (
              <li key={`${item.alias}-${item.model_version}`} className="flex items-center gap-3 text-[13.5px]">
                <span className="w-5 font-mono text-[12px] text-muted-foreground">{index + 1}</span>
                <span className="flex-1">
                  <span className="font-medium">{modelLabel(item.model_name)}</span>{" "}
                  <span className="font-mono text-[11.5px] text-muted-foreground">v{item.model_version}</span>
                  <span className="block text-[12px] text-muted-foreground">{aliasLabel(item.alias)}</span>
                </span>
                <span className={cn("flex items-center gap-1 text-[12.5px]", breaker.tone)}>
                  <breaker.Icon className="size-3.5" aria-hidden />
                  <span className="text-foreground">{breaker.label}</span>
                </span>
              </li>
            );
          })}
        </ol>
      )}
      <p className="mt-3 text-[12px] text-muted-foreground">
        Render’s API holds only the champion and the MLP (DEC-050). The scorer in the cluster runs
        the full chain.
      </p>
    </Panel>
  );
}

function Drift({ summary }: { summary: HealthSummary }) {
  const reports = driftReportsOf(summary);
  return (
    <Panel title="Drift reports">
      {reports.length ? (
        <ul className="divide-y text-[13px]">
          {reports.map((report) => (
            <li key={report.createdAt} className="flex items-center justify-between py-2">
              <time dateTime={report.createdAt}>{absoluteTime(report.createdAt)}</time>
              {report.drifted ? (
                <Chip tone="warn">Drift found</Chip>
              ) : (
                <Chip tone="outline">No drift</Chip>
              )}
            </li>
          ))}
        </ul>
      ) : (
        <p className="text-[13px] text-muted-foreground">
          None yet. The daily monitor compares live edits with the training data.
        </p>
      )}
    </Panel>
  );
}

function RetrainHistory({ runs }: { runs: RetrainRun[] }) {
  return (
    <Panel title="Retrain history">
      {runs.length === 0 ? (
        <EmptyState title="No retrains yet">
          Retraining runs weekly by default, and sooner on drift. Each run’s before vs after report
          shows here.
        </EmptyState>
      ) : (
        <ul className="divide-y">
          {runs.map((run) => {
            const status = runStatus(run.status);
            return (
              <li key={`${run.id}-${run.startedAt}`} className="py-3 text-[13.5px]">
                <div className="flex flex-wrap items-center justify-between gap-2">
                  <span>
                    <span className="font-medium capitalize">{triggerLabel(run.trigger)}</span>{" "}
                    <time dateTime={run.startedAt} className="text-muted-foreground">
                      {absoluteTime(run.startedAt)}
                    </time>
                  </span>
                  <Chip tone={status.tone}>{status.label}</Chip>
                </div>
                <RunNotes run={run} />
                {run.comparison && <BeforeAfter run={run} />}
              </li>
            );
          })}
        </ul>
      )}
    </Panel>
  );
}

const TRIGGERS: Record<string, string> = {
  manual: "Manual retrain",
  manual_promote: "Manual promotion",
  schedule: "Scheduled retrain",
  drift: "Drift",
  performance: "Performance drop",
};

function triggerLabel(trigger: string): string {
  return TRIGGERS[trigger] ?? trigger.replaceAll("_", " ");
}

/** Why a run happened and what came after it. Every line is text the trainer or monitor wrote. */
function RunNotes({ run }: { run: RetrainRun }) {
  const notes: { key: string; text: string; strong?: boolean }[] = [];
  if (run.reasons.length) notes.push({ key: "why", text: `Why: ${run.reasons.join("; ")}.` });
  if (run.forced) notes.push({ key: "forced", text: "A manual override: the gate was skipped, the artifact checks weren’t." });
  if (run.status === "shadow" && run.candidateVersion) {
    notes.push({
      key: "shadow",
      text: `Version ${run.candidateVersion} is scoring live edits with no effect on users${
        run.shadowStartedAt ? ` since ${absoluteTime(run.shadowStartedAt)}` : ""
      }. It’s promoted only if the shadow check passes.`,
    });
  }
  if (run.gateFailures.length) {
    notes.push({ key: "gate", text: `Promotion gate failed: ${run.gateFailures.map((name) => name.replaceAll("_", " ")).join(", ")}.` });
  }
  if (run.postCheck) {
    notes.push({ key: "post", text: `Post-promotion check: ${run.postCheck.reason}.`, strong: run.postCheck.rollback });
  }
  if (run.rolledBack) {
    const { at, reason, by } = run.rolledBack;
    notes.push({
      key: "rollback",
      text: `Rolled back${at ? ` ${absoluteTime(at)}` : ""}${by ? ` by ${by === "monitor" ? "the monitor" : "the owner"}` : ""}${reason ? `: ${reason}` : ""}.`,
      strong: true,
    });
  }
  if (!notes.length) return null;
  return (
    <ul className="mt-1.5 space-y-0.5 text-[12.5px] text-muted-foreground">
      {notes.map((note) => (
        <li key={note.key} className={cn(note.strong && "font-medium text-foreground")}>
          {note.text}
        </li>
      ))}
    </ul>
  );
}

function BeforeAfter({ run }: { run: RetrainRun }) {
  const { before, after, difference, promoted, reason } = run.comparison!;
  const sides = [
    { label: "Before", side: before },
    { label: "After", side: after },
  ];
  return (
    <div className="mt-2 rounded-md bg-muted/60 p-3">
      <table className="w-full text-left text-[13px]">
        <caption className="sr-only">Before vs after on the same test rows</caption>
        <thead className="text-[11px] tracking-[0.08em] text-muted-foreground uppercase">
          <tr>
            <th scope="col" className="font-medium" />
            <th scope="col" className="font-medium">Model</th>
            <th scope="col" className="text-right font-medium">PR-AUC (95% interval)</th>
          </tr>
        </thead>
        <tbody>
          {sides.map(({ label, side }) => (
            <tr key={label}>
              <th scope="row" className="py-1 pr-3 font-medium">{label}</th>
              <td className="py-1">
                {side.model ? modelLabel(side.model) : "—"}
                {side.version && <span className="font-mono text-muted-foreground"> v{side.version}</span>}
              </td>
              <td className="py-1 text-right font-mono tabular">
                {side.prAuc !== null ? side.prAuc.toFixed(4) : <NotMeasured />}
                {side.prAucInterval && (
                  <span className="text-muted-foreground">
                    {" "}({side.prAucInterval[0].toFixed(4)}–{side.prAucInterval[1].toFixed(4)})
                  </span>
                )}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
      {difference && (
        <p className="mt-2 text-[12.5px]">
          Paired difference:{" "}
          <span className="font-mono">
            {difference.mean >= 0 ? "+" : ""}
            {difference.mean.toFixed(4)} ({difference.low.toFixed(4)} to {difference.high.toFixed(4)})
          </span>
          {difference.low > 0 ? " — a clear improvement." : " — the interval includes zero, so not a clear win."}
        </p>
      )}
      {(promoted !== null || reason) && (
        <p className="mt-1 text-[12.5px] text-muted-foreground">
          {promoted === true ? "Promoted." : promoted === false ? "Not promoted." : ""} {reason}
        </p>
      )}
    </div>
  );
}

function Panel({
  title,
  icon,
  children,
}: {
  title: string;
  icon?: React.ReactNode;
  children: React.ReactNode;
}) {
  return (
    <section className="rounded-lg border bg-card p-5">
      <h2 className="mb-3 flex items-center gap-2 font-display text-[19px] font-medium">
        {icon}
        {title}
      </h2>
      {children}
    </section>
  );
}

// --- Owner controls ---

type AdminKind = "retrain" | "promote" | "rollback";

const ADMIN_TEXT: Record<AdminKind, { title: string; help: string; button: string }> = {
  retrain: {
    title: "Retrain now",
    help: "Asks the cluster to build a snapshot and train. Render holds no cluster credentials, so this adds a request that the cluster picks up.",
    button: "Request a retrain",
  },
  promote: {
    title: "Promote a version",
    help: "Makes this version champion, skipping the gate but never the checksum and smoke test. Your reason goes in the audit log.",
    button: "Request promotion",
  },
  rollback: {
    title: "Roll back",
    help: "Puts back the chain from before the newest promotion. The cluster carries it out, and serving picks it up within 5 minutes.",
    button: "Request rollback",
  },
};

function AdminPanel() {
  const seat = useSeat();
  const isOwner = seat.role === "user";
  return (
    <section aria-labelledby="admin-heading" className="rounded-lg border bg-card p-5">
      <h2 id="admin-heading" className="flex items-center gap-2 font-display text-[19px] font-medium">
        <Lock className="size-4" aria-hidden /> Owner controls
      </h2>
      {!isOwner && (
        <p className="mt-1 text-[13px] text-muted-foreground">
          Only the owner’s account can retrain, promote, or roll back. The server refuses these for
          a guest, whatever the page shows.
        </p>
      )}
      <div className="mt-4 grid gap-4 md:grid-cols-3">
        {(Object.keys(ADMIN_TEXT) as AdminKind[]).map((kind) => (
          <AdminForm key={kind} kind={kind} enabled={isOwner} />
        ))}
      </div>
      {isOwner && <RecentRequests />}
    </section>
  );
}

function AdminForm({ kind, enabled }: { kind: AdminKind; enabled: boolean }) {
  const store = useSessionStore();
  const models = useModels();
  const text = ADMIN_TEXT[kind];
  const [reason, setReason] = useState("");
  const [version, setVersion] = useState("");
  const [result, setResult] = useState<{ ok: boolean; message: string } | null>(null);
  const [busy, setBusy] = useState(false);
  const versions = models.rows.map((row) => row.entry);
  const ready = enabled && reason.trim().length >= 3 && (kind !== "promote" || version);

  async function submit(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!ready) return;
    setBusy(true);
    try {
      let reply: AdminRequestInfo;
      if (kind === "retrain") reply = await store.api.requestRetrain(reason.trim());
      else if (kind === "promote") reply = await store.api.requestPromote(version, reason.trim());
      else reply = await store.api.requestRollback(reason.trim());
      setResult({ ok: true, message: `Request ${reply.id} is ${reply.status}.` });
      setReason("");
    } catch (error) {
      setResult({
        ok: false,
        message: error instanceof ApiError ? describeError(error) : "Something went wrong.",
      });
    } finally {
      setBusy(false);
    }
  }

  const id = `admin-${kind}`;
  return (
    <form onSubmit={submit} className="flex flex-col rounded-md border p-4" aria-labelledby={`${id}-title`}>
      <h3 id={`${id}-title`} className="text-sm font-semibold">{text.title}</h3>
      <p className="mt-1 text-[12.5px] text-muted-foreground">{text.help}</p>
      {kind === "promote" && (
        <>
          <label htmlFor={`${id}-version`} className="mt-3 text-[12.5px] font-medium">Version</label>
          <select
            id={`${id}-version`}
            disabled={!enabled}
            value={version}
            onChange={(event) => setVersion(event.target.value)}
            className="mt-1 h-9 rounded-md border bg-background px-2 text-sm disabled:opacity-50"
          >
            <option value="">Choose…</option>
            {versions.map((entry) => (
              <option key={`${entry.model_name}-${entry.model_version}`} value={entry.model_version}>
                {modelLabel(entry.model_name)} v{entry.model_version}
              </option>
            ))}
          </select>
        </>
      )}
      <label htmlFor={`${id}-reason`} className="mt-3 text-[12.5px] font-medium">Reason</label>
      <textarea
        id={`${id}-reason`}
        disabled={!enabled}
        value={reason}
        onChange={(event) => setReason(event.target.value)}
        maxLength={500}
        rows={2}
        placeholder={enabled ? "Recorded in the audit log" : "Owner only"}
        className="mt-1 resize-none rounded-md border bg-background px-2 py-1.5 text-sm disabled:opacity-50"
      />
      <Button type="submit" variant="outline" size="sm" className="mt-3 self-start" disabled={!ready || busy}>
        {busy ? "Sending…" : text.button}
      </Button>
      {result && (
        <p role="status" className={cn("mt-2 text-[12.5px]", result.ok ? "text-diff-added" : "text-risk-flagged")}>
          {result.message}
        </p>
      )}
    </form>
  );
}

function RecentRequests() {
  const store = useSessionStore();
  const requests = usePoll("admin-requests", (signal) => store.api.adminRequests(signal), {
    intervalMs: 30_000,
  });
  const items = requests.data?.items ?? [];
  return (
    <div className="mt-5">
      <h3 className="text-sm font-semibold">Recent requests</h3>
      {items.length === 0 ? (
        <p className="mt-1 text-[13px] text-muted-foreground">None yet.</p>
      ) : (
        <ul className="mt-2 divide-y text-[13px]">
          {items.slice(0, 8).map((item) => (
            <li key={item.id} className="flex flex-wrap items-center justify-between gap-2 py-2">
              <span>
                <span className="font-medium capitalize">{item.kind.replaceAll("_", " ")}</span>{" "}
                <span className="text-muted-foreground">“{item.reason}”</span>
              </span>
              <span className="flex items-center gap-2">
                <time dateTime={item.requested_at} className="text-muted-foreground">
                  {absoluteTime(item.requested_at)}
                </time>
                <Chip tone={item.status === "failed" ? "danger" : item.status === "done" ? "ok" : "outline"}>
                  {item.status}
                </Chip>
              </span>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
