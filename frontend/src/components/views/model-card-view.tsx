"use client";

import Link from "next/link";
import { useParams } from "next/navigation";
import { AlertTriangle, ArrowLeft } from "lucide-react";
import { Chip } from "@/components/ui/chip";
import { Skeleton } from "@/components/ui/skeleton";
import { PrCurvesChart } from "@/components/charts/lazy";
import { IntervalStrip } from "@/components/models/interval-strip";
import { EmptyState, ErrorState } from "@/components/state-notice";
import { CurveTable, Legend, NotMeasured } from "@/components/models/parts";
import { absoluteTime } from "@/lib/format";
import { aliasLabel, modelLabel, seriesColor, sliceLabel, type ModelStats } from "@/lib/models";
import { curveSeries, useModels, type ModelRow } from "@/lib/use-models";

export function ModelCardView() {
  const name = useParams<{ name: string }>().name;
  const models = useModels();
  const row = models.rows.find((item) => item.entry.model_name === name);

  return (
    <div className="mx-auto max-w-6xl px-4 py-8">
      <Link
        href="/models"
        className="inline-flex items-center gap-1.5 rounded-sm text-[13px] font-medium text-muted-foreground hover:text-foreground"
      >
        <ArrowLeft className="size-3.5" aria-hidden /> Models
      </Link>
      {models.status === "loading" && <Skeleton className="mt-6 h-96 w-full" />}
      {models.status === "error" && models.error && (
        <div className="mt-6 rounded-lg border bg-card">
          <ErrorState error={models.error} onRetry={models.refresh} />
        </div>
      )}
      {models.status === "ready" && !row && (
        <div className="mt-6 rounded-lg border bg-card">
          <EmptyState title="No such model">It isn’t registered in this deployment.</EmptyState>
        </div>
      )}
      {row && <Card row={row} all={models.rows} />}
    </div>
  );
}

function Card({ row, all }: { row: ModelRow; all: ModelRow[] }) {
  const { entry, stats } = row;
  const color = seriesColor(entry.model_name);
  const series = curveSeries(all);
  return (
    <article className="mt-5">
      <header className="animate-rise">
        <p className="text-[11px] font-medium tracking-[0.14em] text-risk-flagged uppercase">
          Model card
        </p>
        <h1 className="mt-1.5 flex items-center gap-3 font-display text-[36px] leading-tight font-medium">
          <span aria-hidden className="h-7 w-1.5 rounded-full" style={{ background: color }} />
          {modelLabel(entry.model_name)}
          <span className="font-mono text-base text-muted-foreground">v{entry.model_version}</span>
        </h1>
        <div className="mt-2 flex flex-wrap gap-1.5">
          <Chip tone={entry.alias === "champion" ? "ok" : "outline"}>{aliasLabel(entry.alias)}</Chip>
          <Chip tone="outline">Features v{entry.feature_version}</Chip>
          <Chip tone="outline">Registered {absoluteTime(entry.registered_at)}</Chip>
        </div>
      </header>

      <div className="mt-8 grid gap-6 lg:grid-cols-[minmax(0,1fr)_20rem]">
        <div className="min-w-0 space-y-6">
          <Section title="What it does">
            <p>
              Ranks English Wikipedia article edits by how likely they are to be reverted within 48
              hours, so human patrollers read the riskiest first. It never reverts anything.
            </p>
            <p className="mt-2">
              <strong className="font-semibold">It isn’t a vandalism detector in the strict sense.</strong>{" "}
              A revert also covers good-faith mistakes, and some vandalism is never reverted.
              Don’t use it to judge an editor.
            </p>
          </Section>

          <Section title="Results on the test window">
            <Results stats={stats} color={color} />
          </Section>

          <Section title="Slices">
            <Slices stats={stats} color={color} />
          </Section>

          <Section title="Precision and recall">
            {series.length ? (
              <>
                <Legend names={series.map((s) => s.name)} highlight={entry.model_name} />
                <PrCurvesChart series={series} highlight={entry.model_name} prevalence={stats.curve?.prevalence ?? null} />
                <CurveTable series={series} />
              </>
            ) : (
              <p className="text-muted-foreground">No curve stored for this version.</p>
            )}
          </Section>
        </div>

        <aside className="space-y-4">
          <Facts title="Data">
            <Fact label="Snapshot">{stats.snapshot ?? <NotMeasured />}</Fact>
            <Fact label="Content hash">
              <span className="font-mono text-[11.5px] break-all">{entry.snapshot_hash.slice(0, 16)}…</span>
            </Fact>
            <Fact label="Training rows">{count(stats.trainRows)}</Fact>
            <Fact label="Test rows">{count(stats.testRows)}</Fact>
            <Fact label="Test reverted">{count(stats.testPositives)}</Fact>
          </Facts>
          <Facts title="Build">
            <Fact label="Parameters">{count(stats.parameters)}</Fact>
            <Fact label="Served size">
              {stats.servedBytes !== null ? `${(stats.servedBytes / 1e6).toFixed(1)} MB` : <NotMeasured />}
            </Fact>
            <Fact label="Trained on">{stats.device ?? <NotMeasured />}</Fact>
            <Fact label="Training time">
              {stats.fitMinutes !== null ? `${stats.fitMinutes} min` : <NotMeasured />}
            </Fact>
            <Fact label="ONNX vs PyTorch">
              {stats.parityGap !== null ? `PR-AUC gap ${stats.parityGap.toFixed(5)}` : <NotMeasured />}
            </Fact>
          </Facts>
          <Facts title="Speed">
            <Fact label="Laptop p50">
              {stats.laptopP50Ms !== null ? `${stats.laptopP50Ms.toFixed(1)} ms` : <NotMeasured />}
            </Fact>
            <Fact label="Laptop p95">
              {stats.laptopP95Ms !== null ? `${stats.laptopP95Ms.toFixed(1)} ms` : <NotMeasured />}
            </Fact>
          </Facts>
          {stats.warnings.length > 0 && (
            <section className="rounded-lg border border-risk-elevated/40 bg-risk-elevated-soft p-4 text-[13px]">
              <h2 className="flex items-center gap-1.5 font-semibold">
                <AlertTriangle className="size-4" aria-hidden /> Training warnings
              </h2>
              <ul className="mt-2 list-disc space-y-1 pl-5">
                {stats.warnings.map((warning) => (
                  <li key={warning}>{warning}</li>
                ))}
              </ul>
            </section>
          )}
        </aside>
      </div>
    </article>
  );
}

function Results({ stats, color }: { stats: ModelStats; color: string }) {
  const at = stats.atThreshold;
  return (
    <dl className="grid gap-x-6 gap-y-3 sm:grid-cols-[12rem_1fr]">
      <Metric label="PR-AUC">
        {stats.prAuc !== null ? (
          <IntervalStrip value={stats.prAuc} interval={stats.prAucInterval} max={0.5} color={color} label="PR-AUC" />
        ) : (
          <NotMeasured />
        )}
      </Metric>
      <Metric label="Recall at 90% precision">
        {stats.recallAt90 !== null ? (
          <IntervalStrip value={stats.recallAt90} interval={stats.recallAt90Interval} max={0.2} color={color} label="Recall at 90% precision" />
        ) : (
          <NotMeasured />
        )}
      </Metric>
      <Metric label="Precision in the top 50">{fixed(stats.precisionAt50, 2)}</Metric>
      <Metric label="ROC-AUC">{fixed(stats.rocAuc, 4)}</Metric>
      <Metric label="Calibration error (ECE)">
        {stats.eceBefore !== null && stats.eceAfter !== null ? (
          <span className="font-mono tabular">
            {stats.eceBefore.toFixed(4)} → {stats.eceAfter.toFixed(4)}{" "}
            <span className="font-sans text-muted-foreground">after temperature scaling</span>
          </span>
        ) : (
          fixed(stats.eceAfter, 4)
        )}
      </Metric>
      <Metric label="At its threshold">
        {at.precision !== null && at.recall !== null ? (
          <span>
            <span className="font-mono tabular">
              precision {at.precision.toFixed(3)}, recall {at.recall.toFixed(3)}
            </span>
            <span className="block text-[12px] text-muted-foreground">
              {stats.thresholdRule === "90_percent_precision"
                ? "Threshold set for 90% precision on validation."
                : "Threshold set for the best F1 on validation, because 90% precision wasn’t reachable."}
            </span>
          </span>
        ) : (
          <NotMeasured />
        )}
      </Metric>
      <Metric label="Golden set PR-AUC">{fixed(stats.goldenPrAuc, 4)}</Metric>
    </dl>
  );
}

function Slices({ stats, color }: { stats: ModelStats; color: string }) {
  const entries = Object.entries(stats.slices);
  if (!entries.length) return <p className="text-muted-foreground">No slices stored.</p>;
  return (
    <>
      <table className="w-full text-left text-[13px]">
        <caption className="sr-only">PR-AUC and false positive rate by group of edits</caption>
        <thead className="text-[11px] tracking-[0.08em] text-muted-foreground uppercase">
          <tr>
            <th scope="col" className="py-1.5 font-medium">Group</th>
            <th scope="col" className="py-1.5 font-medium">PR-AUC</th>
            <th scope="col" className="py-1.5 text-right font-medium">Edits</th>
            <th scope="col" className="py-1.5 text-right font-medium">False positive rate</th>
          </tr>
        </thead>
        <tbody className="divide-y">
          {entries.map(([name, slice]) => (
            <tr key={name}>
              <th scope="row" className="py-2 pr-4 font-normal">{sliceLabel(name)}</th>
              <td className="py-2 pr-4">
                {slice.rows > 0 && slice.prAuc !== null ? (
                  <span className="flex items-center gap-2">
                    <span aria-hidden className="h-2 w-28 rounded-full bg-muted">
                      <span
                        className="block h-full rounded-full"
                        style={{ width: `${Math.min(100, slice.prAuc * 100)}%`, background: color }}
                      />
                    </span>
                    <span className="font-mono tabular">{slice.prAuc.toFixed(3)}</span>
                  </span>
                ) : (
                  <NotMeasured />
                )}
              </td>
              <td className="py-2 text-right font-mono tabular">{slice.rows.toLocaleString("en-US")}</td>
              <td className="py-2 text-right font-mono tabular">
                {slice.rows > 0 && slice.falsePositiveRate !== null
                  ? `${(slice.falsePositiveRate * 100).toFixed(3)}%`
                  : "—"}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
      {stats.falsePositiveGap !== null && (
        <p className="mt-2 text-[12.5px] text-muted-foreground">
          False positive gap, logged-out minus registered:{" "}
          <span className="font-mono">{(stats.falsePositiveGap * 100).toFixed(3)}%</span>. Groups
          with no edit count (the backfill) are left out of the experience slices.
        </p>
      )}
    </>
  );
}

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <section className="rounded-lg border bg-card p-5 text-[14px] leading-relaxed">
      <h2 className="mb-3 font-display text-[19px] font-medium">{title}</h2>
      {children}
    </section>
  );
}

function Facts({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <section className="rounded-lg border bg-card p-5">
      <h2 className="text-sm font-semibold">{title}</h2>
      <dl className="mt-3 grid grid-cols-[auto_1fr] gap-x-4 gap-y-1.5 text-[13px]">{children}</dl>
    </section>
  );
}

function Fact({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <>
      <dt className="text-muted-foreground">{label}</dt>
      <dd className="min-w-0 text-right break-words">{children}</dd>
    </>
  );
}

function Metric({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <>
      <dt className="text-[13px] text-muted-foreground sm:pt-0.5">{label}</dt>
      <dd>{children}</dd>
    </>
  );
}

function fixed(value: number | null, digits: number) {
  return value !== null ? <span className="font-mono tabular">{value.toFixed(digits)}</span> : <NotMeasured />;
}

function count(value: number | null) {
  return value !== null ? <span className="font-mono tabular">{value.toLocaleString("en-US")}</span> : <NotMeasured />;
}
