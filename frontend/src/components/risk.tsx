import { cn } from "@/lib/utils";
import { ELEVATED_SCORE, formatScore, RISK_LABEL, riskBand, type RiskBand } from "@/lib/risk";

const TEXT: Record<RiskBand, string> = {
  flagged: "text-risk-flagged",
  elevated: "text-risk-elevated",
  low: "text-risk-low",
  unscored: "text-muted-foreground",
};

export const RISK_BAR: Record<RiskBand, string> = {
  flagged: "bg-risk-flagged",
  elevated: "bg-risk-elevated",
  low: "bg-risk-low/45",
  unscored: "bg-border",
};

/** A different shape for each band, so it reads without color too. */
export function RiskGlyph({ band, className }: { band: RiskBand; className?: string }) {
  const shape = {
    flagged: <path d="M6 1.2 11 10.2H1Z" />,
    elevated: <path d="M6 1 11 6 6 11 1 6Z" />,
    low: <circle cx="6" cy="6" r="3.2" />,
    unscored: <circle cx="6" cy="6" r="3.6" fill="none" stroke="currentColor" strokeWidth="1.4" strokeDasharray="2 1.6" />,
  }[band];
  return (
    <svg viewBox="0 0 12 12" aria-hidden className={cn("size-3 shrink-0 fill-current", className)}>
      {shape}
    </svg>
  );
}

export function RiskBadge({
  score,
  threshold,
  size = "sm",
}: {
  score: number | null | undefined;
  threshold: number | null | undefined;
  size?: "sm" | "lg";
}) {
  const band = riskBand(score, threshold);
  const label = RISK_LABEL[band];
  const value = formatScore(score);
  return (
    <span
      className={cn("inline-flex items-center gap-1.5", TEXT[band])}
      title={band === "unscored" ? "No score yet" : `${label}: ${value} chance of being reverted`}
    >
      <RiskGlyph band={band} className={size === "lg" ? "size-3.5" : undefined} />
      <span
        className={cn(
          "font-medium tracking-wide uppercase",
          size === "lg" ? "text-xs" : "text-[10.5px]",
        )}
      >
        {label}
      </span>
      {band !== "unscored" && (
        <span
          className={cn(
            "font-mono tabular text-foreground",
            size === "lg" ? "text-sm" : "text-[12.5px]",
          )}
        >
          {value}
        </span>
      )}
    </span>
  );
}

/** The score on a 0 to 100% track, with the two band edges marked. */
export function RiskMeter({ score, threshold }: { score: number; threshold: number | null }) {
  const band = riskBand(score, threshold);
  const clamp = (value: number) => Math.min(100, Math.max(0, value * 100));
  return (
    <div>
      <div
        role="meter"
        aria-label="Risk score"
        aria-valuemin={0}
        aria-valuemax={100}
        aria-valuenow={Math.round(score * 100)}
        aria-valuetext={`${formatScore(score)}, ${RISK_LABEL[band]}`}
        className="relative h-2 rounded-full bg-muted"
      >
        <div
          className={cn("absolute inset-y-0 left-0 rounded-full", RISK_BAR[band])}
          style={{ width: `${Math.max(clamp(score), 1.5)}%` }}
        />
        <Tick at={clamp(ELEVATED_SCORE)} />
        {threshold !== null && <Tick at={clamp(threshold)} strong />}
      </div>
      <div className="mt-2 flex justify-between font-mono text-[11px] text-muted-foreground tabular">
        <span>0%</span>
        <span>
          elevated ≥ {formatScore(ELEVATED_SCORE)}
          {threshold !== null && <> · flagged ≥ {formatScore(threshold)}</>}
        </span>
        <span>100%</span>
      </div>
    </div>
  );
}

function Tick({ at, strong }: { at: number; strong?: boolean }) {
  return (
    <span
      aria-hidden
      className={cn(
        "absolute -top-1 -bottom-1 w-px",
        strong ? "bg-risk-flagged" : "bg-muted-foreground/60",
      )}
      style={{ left: `${at}%` }}
    />
  );
}
