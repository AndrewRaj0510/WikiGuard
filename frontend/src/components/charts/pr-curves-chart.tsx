"use client";

import {
  CartesianGrid,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { modelLabel, seriesColor } from "@/lib/models";

export interface CurveSeries {
  name: string;
  /** Precision at each whole percent of recall, from 1% to 100%. */
  precision: number[];
}

/**
 * Precision against recall for several models on the same test edits. Loaded on demand
 * (charts/lazy.tsx), so Recharts never weighs on a page's first load.
 *
 * Every curve shares one x grid, a point per whole percent of recall, so one crosshair reads all
 * of them at once. The highlighted model is drawn heavier; identity is also in the legend and the
 * tooltip, never color alone.
 */
export default function PrCurvesChart({
  series,
  highlight,
  prevalence,
}: {
  series: CurveSeries[];
  highlight: string | null;
  prevalence: number | null;
}) {
  const rows = Array.from({ length: 100 }, (_, index) => {
    const row: Record<string, number> = { recall: (index + 1) / 100 };
    for (const item of series) {
      const value = item.precision[index];
      if (value !== undefined) row[item.name] = value;
    }
    return row;
  });
  return (
    <ResponsiveContainer width="100%" height={320}>
      <LineChart data={rows} margin={{ top: 8, right: 16, bottom: 20, left: 4 }}>
        <CartesianGrid stroke="var(--border)" strokeDasharray="0" vertical={false} />
        <XAxis
          dataKey="recall"
          type="number"
          domain={[0, 1]}
          ticks={[0, 0.25, 0.5, 0.75, 1]}
          tickFormatter={percent}
          stroke="var(--muted-foreground)"
          tick={{ fontSize: 11, fill: "var(--muted-foreground)" }}
          tickLine={false}
          axisLine={{ stroke: "var(--border)" }}
          label={{
            value: "Recall",
            position: "insideBottom",
            offset: -12,
            fill: "var(--muted-foreground)",
            fontSize: 11,
          }}
        />
        <YAxis
          type="number"
          domain={[0, 1]}
          ticks={[0, 0.25, 0.5, 0.75, 1]}
          tickFormatter={percent}
          tick={{ fontSize: 11, fill: "var(--muted-foreground)" }}
          tickLine={false}
          axisLine={false}
          width={40}
        />
        {prevalence !== null && (
          <ReferenceLine
            y={prevalence}
            stroke="var(--muted-foreground)"
            strokeDasharray="4 4"
            label={{
              value: "Constant score",
              position: "insideTopRight",
              fill: "var(--muted-foreground)",
              fontSize: 11,
            }}
          />
        )}
        <Tooltip
          cursor={{ stroke: "var(--muted-foreground)", strokeWidth: 1 }}
          content={<CurveTooltip highlight={highlight} />}
        />
        {series.map((item) => (
          <Line
            key={item.name}
            dataKey={item.name}
            name={modelLabel(item.name)}
            type="linear"
            stroke={seriesColor(item.name)}
            strokeWidth={item.name === highlight ? 3 : 1.75}
            strokeOpacity={highlight && item.name !== highlight ? 0.75 : 1}
            dot={false}
            activeDot={{ r: 4, stroke: "var(--card)", strokeWidth: 2 }}
            isAnimationActive={false}
          />
        ))}
      </LineChart>
    </ResponsiveContainer>
  );
}

function percent(value: number): string {
  return `${Math.round(value * 100)}%`;
}

interface TooltipEntry {
  dataKey?: string | number;
  value?: number | string;
}

function CurveTooltip({
  active,
  payload,
  label,
  highlight,
}: {
  active?: boolean;
  payload?: TooltipEntry[];
  label?: number;
  highlight: string | null;
}) {
  if (!active || !payload?.length) return null;
  return (
    <div className="rounded-md border bg-card px-3 py-2 text-[12px] shadow-md">
      <p className="mb-1 font-medium">Recall {percent(Number(label))}</p>
      <ul className="space-y-0.5">
        {payload.map((entry) => {
          const name = String(entry.dataKey);
          return (
            <li key={name} className="flex items-center justify-between gap-4">
              <span className="flex items-center gap-1.5">
                <span
                  aria-hidden
                  className="inline-block h-0.5 w-3 rounded-full"
                  style={{ background: seriesColor(name) }}
                />
                <span className={name === highlight ? "font-semibold" : undefined}>
                  {modelLabel(name)}
                </span>
              </span>
              <span className="font-mono tabular">
                {typeof entry.value === "number" ? percent(entry.value) : "—"}
              </span>
            </li>
          );
        })}
      </ul>
      <p className="mt-1 text-muted-foreground">precision</p>
    </div>
  );
}
