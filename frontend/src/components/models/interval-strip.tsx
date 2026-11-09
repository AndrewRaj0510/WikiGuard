import type { Interval } from "@/lib/models";

/**
 * A value with its 95% interval: a whisker for the interval and a dot for the value, on a scale
 * shared by every row, so the strips compare at a glance. The numbers are always printed as well.
 */
export function IntervalStrip({
  value,
  interval,
  max,
  color,
  label,
}: {
  value: number;
  interval: Interval | null;
  max: number;
  color: string;
  label: string;
}) {
  const at = (x: number) => `${Math.min(100, Math.max(0, (x / max) * 100))}%`;
  return (
    <span className="flex items-center gap-3">
      <span
        className="relative h-4 w-20 shrink-0 lg:w-24"
        role="img"
        aria-label={
          interval
            ? `${label} ${value.toFixed(3)}, 95% interval ${interval[0].toFixed(3)} to ${interval[1].toFixed(3)}`
            : `${label} ${value.toFixed(3)}`
        }
      >
        <span aria-hidden className="absolute inset-x-0 top-1/2 h-px bg-border" />
        {interval && (
          <span
            aria-hidden
            className="absolute top-1/2 h-0.5 -translate-y-1/2 rounded-full"
            style={{ left: at(interval[0]), width: `calc(${at(interval[1])} - ${at(interval[0])})`, background: color }}
          />
        )}
        <span
          aria-hidden
          className="absolute top-1/2 size-2.5 -translate-x-1/2 -translate-y-1/2 rounded-full ring-2 ring-card"
          style={{ left: at(value), background: color }}
        />
      </span>
      <span className="font-mono leading-tight whitespace-nowrap tabular">
        <span className="block text-[12.5px]">{value.toFixed(3)}</span>
        {interval && (
          <span className="block text-[11px] text-muted-foreground">
            {interval[0].toFixed(3)}–{interval[1].toFixed(3)}
          </span>
        )}
      </span>
    </span>
  );
}
