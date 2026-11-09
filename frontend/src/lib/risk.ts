/**
 * How a score is shown. The score is the model's calibrated chance that the edit gets reverted.
 *
 * - Flagged: at or above the scoring model's own threshold, which sits at 90% precision on
 *   validation. It's a short, high-confidence list: about 3% recall on test (docs/metrics.md).
 * - Elevated: 0.18 or more, where the top 5% of RoBERTa's calibrated test scores starts
 *   (DEC-061). It's tied to RoBERTa's calibration, so it's rougher when a fallback scored the edit.
 * - Low: everything else. The queue's order, not the band, is what does the real work (F-083).
 *
 * The band is always shown as a word and a shape as well as a color (WCAG 1.4.1).
 */
export const ELEVATED_SCORE = 0.18;

export type RiskBand = "flagged" | "elevated" | "low" | "unscored";

export const RISK_LABEL: Record<RiskBand, string> = {
  flagged: "Flagged",
  elevated: "Elevated",
  low: "Low",
  unscored: "Not scored",
};

export function riskBand(
  score: number | null | undefined,
  threshold: number | null | undefined,
): RiskBand {
  if (score === null || score === undefined || !Number.isFinite(score)) return "unscored";
  if (threshold !== null && threshold !== undefined && score >= threshold) return "flagged";
  if (score >= ELEVATED_SCORE) return "elevated";
  return "low";
}

/** A score as a percentage. The ends say "<1%" and ">99%", since a model is never certain. */
export function formatScore(score: number | null | undefined): string {
  if (score === null || score === undefined || !Number.isFinite(score)) return "—";
  const percent = score * 100;
  if (percent > 0 && percent < 1) return "<1%";
  if (percent > 99 && percent < 100) return ">99%";
  return `${Math.round(percent)}%`;
}
