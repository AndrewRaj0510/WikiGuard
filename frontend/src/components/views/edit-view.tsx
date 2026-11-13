"use client";

import { useState } from "react";
import Link from "next/link";
import { useParams, useRouter } from "next/navigation";
import {
  AlertTriangle,
  ArrowLeft,
  ArrowRight,
  ArrowUpRight,
  EyeOff,
  Feather,
  LifeBuoy,
  UserRoundX,
} from "lucide-react";
import { Chip } from "@/components/ui/chip";
import { Skeleton } from "@/components/ui/skeleton";
import { DiffView } from "@/components/diff-view";
import { ExplainPanel } from "@/components/explain-panel";
import { ReviewControls } from "@/components/review-controls";
import { RiskBadge, RiskMeter } from "@/components/risk";
import { EmptyState, ErrorState } from "@/components/state-notice";
import type { EditDetail, ScoreInfo } from "@/lib/api/types";
import { absoluteTime, formatBytesDelta } from "@/lib/format";
import { formatScore } from "@/lib/risk";
import { listOrder, neighbour, type ListOrder } from "@/lib/list-order";
import { useSessionStore } from "@/lib/session/provider";
import { useShortcuts } from "@/lib/shortcuts";
import { usePoll } from "@/lib/use-poll";
import { articleUrl, diffUrl, EXTERNAL_REL, parseRevId } from "@/lib/wikipedia";
import { cn } from "@/lib/utils";

// A detail changes rarely: a rescore, a review, or its label after 48 hours.
const DETAIL_POLL_MS = 30_000;

export function EditView() {
  const revId = parseRevId(useParams<{ revId: string }>().revId);
  if (revId === null) {
    return (
      <Frame>
        <EmptyState title="That isn’t an edit ID">Revision IDs are whole numbers.</EmptyState>
      </Frame>
    );
  }
  return <EditLoader revId={revId} />;
}

function EditLoader({ revId }: { revId: number }) {
  const store = useSessionStore();
  const edit = usePoll(`edit:${revId}`, (signal) => store.api.edit(revId, signal), {
    intervalMs: DETAIL_POLL_MS,
  });
  if (edit.status === "loading") return <EditSkeleton />;
  if (edit.status === "error" && edit.error) {
    return (
      <Frame>
        {edit.error.code === "edit_not_found" ? (
          <EmptyState title="No such edit">
            This edit isn’t in WikiGuard’s database: it may have been removed, or it was never
            scored.
          </EmptyState>
        ) : (
          <ErrorState error={edit.error} onRetry={edit.refresh} />
        )}
      </Frame>
    );
  }
  return edit.data ? <EditBody edit={edit.data} onChanged={edit.refresh} /> : null;
}

function Frame({ children }: { children: React.ReactNode }) {
  const [order] = useState(listOrder);
  return (
    <div className="mx-auto max-w-6xl px-4 py-8">
      <BackLink order={order} />
      <div className="mt-6 rounded-lg border bg-card">{children}</div>
    </div>
  );
}

function BackLink({ order }: { order: ListOrder }) {
  return (
    <Link
      href={order.href}
      className="inline-flex items-center gap-1.5 rounded-sm text-[13px] font-medium text-muted-foreground transition-colors hover:text-foreground"
    >
      <ArrowLeft className="size-3.5" aria-hidden /> {order.label}
    </Link>
  );
}

/** Previous and next in the list the patroller came from, as links and as p and n. */
function Stepper({ revId, order }: { revId: number; order: ListOrder }) {
  const router = useRouter();
  const previous = neighbour(revId, -1, order);
  const next = neighbour(revId, 1, order);
  useShortcuts({
    n: () => next !== null && router.push(`/edits/${next}`),
    p: () => previous !== null && router.push(`/edits/${previous}`),
    b: () => router.push(order.href),
  });
  if (previous === null && next === null) return null;
  return (
    <nav aria-label={`Step through the ${order.label.toLowerCase()}`} className="flex items-center gap-1">
      {previous !== null ? (
        <Link
          href={`/edits/${previous}`}
          aria-keyshortcuts="p"
          className="inline-flex items-center gap-1 rounded-md px-2 py-1 text-[13px] font-medium text-muted-foreground hover:bg-accent hover:text-foreground"
        >
          <ArrowLeft className="size-3.5" aria-hidden /> Previous
        </Link>
      ) : null}
      {next !== null ? (
        <Link
          href={`/edits/${next}`}
          aria-keyshortcuts="n"
          className="inline-flex items-center gap-1 rounded-md px-2 py-1 text-[13px] font-medium text-muted-foreground hover:bg-accent hover:text-foreground"
        >
          Next <ArrowRight className="size-3.5" aria-hidden />
        </Link>
      ) : null}
    </nav>
  );
}

function EditBody({ edit, onChanged }: { edit: EditDetail; onChanged: () => void }) {
  const latest = edit.scores[0];
  const [order] = useState(listOrder);
  // The saved decision shows at once; the next poll confirms it from the server.
  const [decision, setDecision] = useState(edit.decision);
  return (
    <article className="mx-auto max-w-6xl px-4 py-8">
      <div className="flex items-center justify-between gap-3">
        <BackLink order={order} />
        <Stepper revId={edit.rev_id} order={order} />
      </div>

      <header className="mt-5 animate-rise">
        <RiskBadge score={edit.score} threshold={edit.threshold} size="lg" />
        <h1 className="mt-2 font-display text-[36px] leading-[1.08] font-medium tracking-[-0.015em] break-words">
          {edit.title}
        </h1>
        <div className="mt-3 flex flex-wrap items-center gap-1.5">
          {edit.is_anonymous && (
            <Chip tone="outline">
              <UserRoundX aria-hidden /> Logged out
            </Chip>
          )}
          {edit.is_minor && (
            <Chip tone="outline">
              <Feather aria-hidden /> Minor edit
            </Chip>
          )}
          {edit.used_fallback && (
            <Chip tone="warn">
              <LifeBuoy aria-hidden /> Scored by a fallback
            </Chip>
          )}
          {edit.content_hidden && (
            <Chip>
              <EyeOff aria-hidden /> Hidden by Wikipedia
            </Chip>
          )}
          <span className="mx-1 text-border" aria-hidden>
            |
          </span>
          <ExternalLink href={diffUrl(edit.rev_id)}>Diff on Wikipedia</ExternalLink>
          <ExternalLink href={articleUrl(edit.title)}>Article</ExternalLink>
        </div>
      </header>

      <div className="mt-8 grid gap-6 lg:grid-cols-[minmax(0,1fr)_21rem]">
        <div className="min-w-0 space-y-4">
          <section aria-labelledby="summary-heading" className="rounded-lg border bg-card px-5 py-4">
            <h2
              id="summary-heading"
              className="text-[11px] font-medium tracking-[0.1em] text-muted-foreground uppercase"
            >
              Edit summary
            </h2>
            <p className="mt-1.5 text-[15px] break-words">
              {edit.comment_hidden ? (
                <span className="text-muted-foreground italic">Hidden by Wikipedia</span>
              ) : edit.comment ? (
                edit.comment
              ) : (
                <span className="text-muted-foreground italic">No edit summary</span>
              )}
            </p>
          </section>
          {!edit.content_hidden && <ContentWarning />}
          <DiffView
            removed={edit.removed_text}
            added={edit.added_text}
            hidden={edit.content_hidden}
          />
        </div>

        <aside className="space-y-4">
          <ReviewControls
            revId={edit.rev_id}
            current={decision}
            onSaved={(saved) => {
              setDecision(saved);
              onChanged();
            }}
          />
          <ScoreCard edit={edit} latest={latest} />
          {!edit.content_hidden && (
            <ExplainPanel revId={edit.rev_id} hasSaved={edit.has_explanation} />
          )}
          <OutcomeCard edit={{ ...edit, decision }} />
          <FactsCard edit={edit} />
        </aside>
      </div>
    </article>
  );
}

/** Vandals write this text, and it can be offensive (PLAN section 8). Said before it's shown. */
function ContentWarning() {
  return (
    <p className="flex items-start gap-2 rounded-md border border-risk-elevated/35 bg-risk-elevated-soft px-3 py-2 text-[12.5px]">
      <AlertTriangle className="mt-0.5 size-3.5 shrink-0" aria-hidden />
      <span>
        <span className="font-semibold">Content warning.</span> This is the raw text of a Wikipedia
        edit. Vandalism can include offensive language, and WikiGuard shows it unedited so a person
        can judge it.
      </span>
    </p>
  );
}

function ExternalLink({ href, children }: { href: string; children: React.ReactNode }) {
  return (
    <a
      href={href}
      target="_blank"
      rel={EXTERNAL_REL}
      className="inline-flex items-center gap-0.5 rounded-sm text-[13px] font-medium underline decoration-border underline-offset-4 transition-colors hover:decoration-foreground"
    >
      {children}
      <ArrowUpRight className="size-3.5" aria-hidden />
      <span className="sr-only">(opens in a new tab)</span>
    </a>
  );
}

function Card({
  title,
  children,
  className,
}: {
  title: string;
  children: React.ReactNode;
  className?: string;
}) {
  return (
    <section className={cn("rounded-lg border bg-card p-5", className)}>
      <h2 className="text-sm font-semibold">{title}</h2>
      <div className="mt-3">{children}</div>
    </section>
  );
}

function ScoreCard({ edit, latest }: { edit: EditDetail; latest: ScoreInfo | undefined }) {
  if (edit.score === null || !latest) {
    return (
      <Card title="Risk">
        <p className="text-[13.5px] text-muted-foreground">
          Not scored yet. The scorer rescores missed edits every minute.
        </p>
      </Card>
    );
  }
  return (
    <Card title="Risk">
      <p className="font-mono text-[40px] leading-none font-medium tracking-tight tabular">
        {formatScore(edit.score)}
      </p>
      <p className="mt-1.5 text-[12.5px] text-muted-foreground">
        chance of being reverted, as the model estimates it
      </p>
      <div className="mt-4">
        <RiskMeter score={edit.score} threshold={edit.threshold} />
      </div>
      <dl className="mt-4 grid grid-cols-[auto_1fr] gap-x-4 gap-y-1.5 text-[13px]">
        <Fact label="Model">
          <span className="font-mono">
            {latest.model_name} v{latest.model_version}
          </span>
        </Fact>
        <Fact label="Role">{latest.alias ?? "—"}</Fact>
        {latest.latency_ms !== null && (
          <Fact label="Latency">
            <span className="font-mono tabular">{latest.latency_ms.toFixed(0)} ms</span>
          </Fact>
        )}
        <Fact label="Scored">
          <time dateTime={latest.scored_at}>{absoluteTime(latest.scored_at)}</time>
        </Fact>
      </dl>
      {(latest.used_fallback || latest.load_shed || latest.forced) && (
        <p className="mt-3 rounded-md bg-risk-elevated-soft px-3 py-2 text-[12.5px] text-foreground">
          {latest.forced
            ? "An operator forced this model with the kill switch."
            : latest.load_shed
              ? "The scorer was behind, so a faster model scored this edit."
              : "The champion didn’t answer in time, so a fallback scored this edit."}
        </p>
      )}
      {edit.scores.length > 1 && (
        <details className="mt-3 text-[12.5px]">
          <summary className="cursor-pointer text-muted-foreground hover:text-foreground">
            {edit.scores.length} scores in all
          </summary>
          <ul className="mt-2 space-y-1">
            {edit.scores.map((score) => (
              <li key={`${score.model_version}-${score.scored_at}`} className="flex justify-between gap-2 font-mono">
                <span>
                  {score.model_name} v{score.model_version}
                </span>
                <span className="tabular">{formatScore(score.score)}</span>
              </li>
            ))}
          </ul>
        </details>
      )}
    </Card>
  );
}

const DECISION_TEXT = {
  vandalism: "Vandalism",
  good_faith_mistake: "Good-faith mistake",
  ok: "OK",
  skip: "Skipped",
} as const;

function OutcomeCard({ edit }: { edit: EditDetail }) {
  const label =
    edit.label === 1
      ? { text: "Reverted within 48 hours", tone: "danger" as const }
      : edit.label === 0
        ? { text: "Still standing after 48 hours", tone: "ok" as const }
        : null;
  return (
    <Card title="Outcome">
      <dl className="grid grid-cols-[auto_1fr] gap-x-4 gap-y-2 text-[13px]">
        <Fact label="Label">
          {label ? (
            <Chip tone={label.tone}>{label.text}</Chip>
          ) : edit.label_excluded_reason ? (
            <span className="text-muted-foreground">Left out: {edit.label_excluded_reason}</span>
          ) : (
            <span className="text-muted-foreground">Pending: labels are final after 48 hours</span>
          )}
        </Fact>
        <Fact label="Review">
          {edit.decision ? DECISION_TEXT[edit.decision] : <span className="text-muted-foreground">Not reviewed</span>}
        </Fact>
      </dl>
    </Card>
  );
}

function FactsCard({ edit }: { edit: EditDetail }) {
  return (
    <Card title="Edit">
      <dl className="grid grid-cols-[auto_1fr] gap-x-4 gap-y-1.5 text-[13px]">
        <Fact label="Revision">
          <span className="font-mono tabular">{edit.rev_id}</span>
        </Fact>
        {edit.parent_rev_id !== null && (
          <Fact label="Parent">
            <span className="font-mono tabular">{edit.parent_rev_id}</span>
          </Fact>
        )}
        <Fact label="Edited">
          <time dateTime={edit.edited_at}>{absoluteTime(edit.edited_at)}</time>
        </Fact>
        <Fact label="Size">
          <span className="font-mono tabular">{formatBytesDelta(edit.bytes_delta)} bytes</span>
        </Fact>
        <Fact label="Editor">
          {edit.user_hidden
            ? "Hidden by Wikipedia"
            : edit.is_anonymous
              ? "Logged out"
              : edit.user_edit_count !== null
                ? `Registered, ${edit.user_edit_count.toLocaleString("en-US")} edits`
                : "Registered"}
        </Fact>
      </dl>
    </Card>
  );
}

function Fact({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <>
      <dt className="text-muted-foreground">{label}</dt>
      <dd className="min-w-0 break-words">{children}</dd>
    </>
  );
}

export function EditSkeleton() {
  return (
    <div aria-busy="true" aria-label="Loading the edit" className="mx-auto max-w-6xl px-4 py-8">
      <Skeleton className="h-4 w-28" />
      <Skeleton className="mt-6 h-4 w-24" />
      <Skeleton className="mt-3 h-10 w-2/3" />
      <div className="mt-8 grid gap-6 lg:grid-cols-[minmax(0,1fr)_21rem]">
        <div className="space-y-4">
          <Skeleton className="h-20" />
          <div className="grid gap-4 md:grid-cols-2">
            <Skeleton className="h-64" />
            <Skeleton className="h-64" />
          </div>
        </div>
        <div className="space-y-4">
          <Skeleton className="h-56" />
          <Skeleton className="h-32" />
        </div>
      </div>
    </div>
  );
}
