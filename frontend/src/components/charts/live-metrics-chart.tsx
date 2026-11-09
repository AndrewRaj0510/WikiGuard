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

export interface LivePoint {
  day: string;
  prAuc: number;
}

/**
 * Live PR-AUC by day, from labels that arrive 48 hours after each edit, against the champion's
 * offline test PR-AUC as a reference line. One series, so its title names it and no legend box is
 * needed. Loaded on demand.
 */
export default function LiveMetricsChart({
  points,
  offline,
}: {
  points: LivePoint[];
  offline: number | null;
}) {
  const top = Math.max(0.5, ...points.map((p) => p.prAuc), offline ?? 0);
  return (
    <ResponsiveContainer width="100%" height={240}>
      <LineChart data={points} margin={{ top: 8, right: 16, bottom: 4, left: 4 }}>
        <CartesianGrid stroke="var(--border)" vertical={false} />
        <XAxis
          dataKey="day"
          tick={{ fontSize: 11, fill: "var(--muted-foreground)" }}
          tickLine={false}
          axisLine={{ stroke: "var(--border)" }}
        />
        <YAxis
          domain={[0, Math.min(1, Math.ceil(top * 10) / 10)]}
          tickFormatter={(value: number) => value.toFixed(2)}
          tick={{ fontSize: 11, fill: "var(--muted-foreground)" }}
          tickLine={false}
          axisLine={false}
          width={40}
        />
        {offline !== null && (
          <ReferenceLine
            y={offline}
            stroke="var(--muted-foreground)"
            strokeDasharray="4 4"
            label={{
              value: `Offline test ${offline.toFixed(3)}`,
              position: "insideTopRight",
              fill: "var(--muted-foreground)",
              fontSize: 11,
            }}
          />
        )}
        <Tooltip
          cursor={{ stroke: "var(--muted-foreground)", strokeWidth: 1 }}
          formatter={(value) => [Number(value).toFixed(3), "Live PR-AUC"]}
          contentStyle={{
            background: "var(--card)",
            border: "1px solid var(--border)",
            borderRadius: 6,
            fontSize: 12,
          }}
        />
        <Line
          dataKey="prAuc"
          name="Live PR-AUC"
          stroke="var(--series-roberta)"
          strokeWidth={2}
          dot={{ r: 3, strokeWidth: 0, fill: "var(--series-roberta)" }}
          activeDot={{ r: 4, stroke: "var(--card)", strokeWidth: 2 }}
          isAnimationActive={false}
        />
      </LineChart>
    </ResponsiveContainer>
  );
}
