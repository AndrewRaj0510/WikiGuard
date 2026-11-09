"use client";

import Link from "next/link";
import { AlertTriangle } from "lucide-react";
import { Chip } from "@/components/ui/chip";
import { PrCurvesChart } from "@/components/charts/lazy";
import { IntervalStrip } from "@/components/models/interval-strip";
import { EmptyState, ErrorState } from "@/components/state-notice";
import { Skeleton } from "@/components/ui/skeleton";
import { PageHeader } from "@/components/views/page-header";
import { CurveTable, Legend, NotMeasured } from "@/components/models/parts";
import { aliasLabel, modelLabel, seriesColor } from "@/lib/models";
import { curveSeries, useModels, type ModelRow } from "@/lib/use-models";

export function ModelsView() {
  const models = useModels();
  const rows = models.rows;
  const champion = rows.find((row) => row.entry.alias === "champion") ?? null;
  const test = champion?.stats;

  return (
    <div className="mx-auto max-w-6xl px-4 py-8">
      <PageHeader
        eyebrow="Evaluation"
        title="Models"
        lede={
          test?.testRows && test.testPositives
            ? `Five neural networks, trained on the same snapshot and measured on the same held-out week: ${test.testRows.toLocaleString("en-US")} edits, ${test.testPositives.toLocaleString("en-US")} of them reverted. The champion and its fallbacks are chosen by these results, never by hand.`
            : "Five neural networks measured on the same held-out week of edits. The champion and its fallbacks are chosen by these results, never by hand."
        }
      />

      {models.status === "loading" && <Skeleton className="mt-8 h-72 w-full" />}
      {models.status === "error" && models.error && (
        <div className="mt-8 rounded-lg border bg-card">
          <ErrorState error={models.error} onRetry={models.refresh} />
        </div>
      )}
      {models.status === "ready" && rows.length === 0 && (
        <div className="mt-8 rounded-lg border bg-card">
          <EmptyState title="No models registered yet">
            The trainer copies each registered model’s results here.
          </EmptyState>
        </div>
      )}
      {rows.length > 0 && (
        <>
          <Leaderboard rows={rows} />
          <CurvesCard rows={rows} champion={champion?.entry.model_name ?? null} />
        </>
      )}
    </div>
  );
}

function Leaderboard({ rows }: { rows: ModelRow[] }) {
  const prMax = niceMax(rows.map((row) => row.stats.prAucInterval?.[1] ?? row.stats.prAuc ?? 0));
  const recallMax = niceMax(
    rows.map((row) => row.stats.recallAt90Interval?.[1] ?? row.stats.recallAt90 ?? 0),
  );
  return (
    <section aria-labelledby="leaderboard-heading" className="mt-8">
      <div className="flex items-baseline justify-between gap-4">
        <h2 id="leaderboard-heading" className="font-display text-[22px] font-medium">
          Leaderboard
        </h2>
        <p className="text-[12.5px] text-muted-foreground">
          Served int8 ONNX files · 95% bootstrap intervals
        </p>
      </div>
      <div className="mt-3 overflow-x-auto rounded-lg border bg-card">
        <table className="w-full min-w-[820px] text-left text-[13px] [&_td]:whitespace-nowrap">
          <caption className="sr-only">
            Each registered model’s results on the test window, in serving order.
          </caption>
          <thead className="border-b text-[11px] tracking-[0.08em] text-muted-foreground uppercase">
            <tr>
              <th scope="col" className="px-4 py-2.5 font-medium">Model</th>
              <th scope="col" className="px-4 py-2.5 font-medium">PR-AUC</th>
              <th scope="col" className="px-4 py-2.5 font-medium">Recall at 90% precision</th>
              <th scope="col" className="px-4 py-2.5 text-right font-medium">Precision, top 50</th>
              <th scope="col" className="px-4 py-2.5 text-right font-medium">ROC-AUC</th>
              <th scope="col" className="px-4 py-2.5 text-right font-medium">ECE</th>
              <th scope="col" className="px-4 py-2.5 text-right font-medium">Size</th>
              <th scope="col" className="px-4 py-2.5 text-right font-medium">Laptop p50 / p95</th>
            </tr>
          </thead>
          <tbody className="divide-y">
            {rows.map(({ entry, stats }) => {
              const color = seriesColor(entry.model_name);
              return (
                <tr key={`${entry.model_name}-${entry.model_version}`} className="hover:bg-accent/50">
                  <th scope="row" className="px-4 py-3 font-normal">
                    <span className="flex items-center gap-2.5">
                      <span aria-hidden className="h-3 w-1 rounded-full" style={{ background: color }} />
                      <Link
                        href={`/models/${encodeURIComponent(entry.model_name)}`}
                        className="font-display text-[16px] font-medium underline-offset-4 hover:underline"
                      >
                        {modelLabel(entry.model_name)}
                      </Link>
                      <span className="font-mono text-[11.5px] text-muted-foreground">
                        v{entry.model_version}
                      </span>
                    </span>
                    <span className="mt-1 flex flex-wrap gap-1 pl-3.5">
                      <Chip tone={entry.alias === "champion" ? "ok" : "outline"}>
                        {aliasLabel(entry.alias)}
                      </Chip>
                      {stats.warnings.length > 0 && (
                        <Chip tone="warn" title={stats.warnings.join(" ")}>
                          <AlertTriangle aria-hidden /> {stats.warnings.length} warning
                          {stats.warnings.length > 1 ? "s" : ""}
                        </Chip>
                      )}
                    </span>
                  </th>
                  <td className="px-4 py-3">
                    {stats.prAuc !== null ? (
                      <IntervalStrip value={stats.prAuc} interval={stats.prAucInterval} max={prMax} color={color} label="PR-AUC" />
                    ) : (
                      <NotMeasured />
                    )}
                  </td>
                  <td className="px-4 py-3">
                    {stats.recallAt90 !== null ? (
                      <IntervalStrip value={stats.recallAt90} interval={stats.recallAt90Interval} max={recallMax} color={color} label="Recall at 90% precision" />
                    ) : (
                      <NotMeasured />
                    )}
                  </td>
                  <Num value={stats.precisionAt50} digits={2} />
                  <Num value={stats.rocAuc} digits={3} />
                  <Num value={stats.eceAfter} digits={4} />
                  <td className="px-4 py-3 text-right font-mono tabular">
                    {stats.servedBytes !== null ? `${(stats.servedBytes / 1e6).toFixed(1)} MB` : <NotMeasured />}
                  </td>
                  <td className="px-4 py-3 text-right font-mono tabular">
                    {stats.laptopP50Ms !== null && stats.laptopP95Ms !== null ? (
                      `${stats.laptopP50Ms.toFixed(1)} / ${stats.laptopP95Ms.toFixed(1)} ms`
                    ) : (
                      <NotMeasured />
                    )}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      <p className="mt-2 text-[12px] text-muted-foreground">
        PR-AUC is the main measure, because only about 6% of edits are reverted. ECE is calibration
        error after temperature scaling: lower is better. Latency is one edit at a time on one
        laptop thread, not on the server.
      </p>
    </section>
  );
}

function CurvesCard({ rows, champion }: { rows: ModelRow[]; champion: string | null }) {
  const series = curveSeries(rows);
  const curveRows = rows.find((row) => row.stats.curve)?.stats.curve;
  return (
    <section aria-labelledby="curves-heading" className="mt-10">
      <h2 id="curves-heading" className="font-display text-[22px] font-medium">
        Precision and recall
      </h2>
      <p className="mt-1 max-w-2xl text-[13.5px] text-muted-foreground">
        How precise each model is when it catches a given share of reverted edits. Higher and
        further right is better.{" "}
        {curveRows?.rows && curveRows.rowsScored
          ? `Drawn on ${curveRows.rows.toLocaleString("en-US")} of the ${curveRows.rowsScored.toLocaleString("en-US")} test edits: a few were removed from the snapshot after training.`
          : ""}
      </p>
      <div className="mt-4 rounded-lg border bg-card p-4">
        {series.length === 0 ? (
          <EmptyState title="No curves stored">
            Run scripts/export_pr_curves.py, then sync the model versions.
          </EmptyState>
        ) : (
          <>
            <Legend names={series.map((s) => s.name)} highlight={champion} />
            <PrCurvesChart series={series} highlight={champion} prevalence={curveRows?.prevalence ?? null} />
            <CurveTable series={series} />
          </>
        )}
      </div>
    </section>
  );
}

function Num({ value, digits }: { value: number | null; digits: number }) {
  return (
    <td className="px-4 py-3 text-right font-mono tabular">
      {value !== null ? value.toFixed(digits) : <NotMeasured />}
    </td>
  );
}


/** A shared scale end, rounded up to a tidy 0.05, so every strip in a column lines up. */
function niceMax(values: number[]): number {
  const top = Math.max(0.05, ...values);
  return Math.ceil(top * 20) / 20;
}

