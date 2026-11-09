const MINUS = "\u2212"; // A real minus sign: it lines up with "+" in a monospace column.

export function formatBytesDelta(delta: number | null | undefined): string {
  if (delta === null || delta === undefined) return "—";
  const size = Math.abs(delta).toLocaleString("en-US");
  if (delta > 0) return `+${size}`;
  if (delta < 0) return `${MINUS}${size}`;
  return "0";
}

/** "now", "42s", "7m", "3h", "2d": short enough for a dense column. */
export function relativeTime(iso: string, now: number): string {
  const then = Date.parse(iso);
  if (!Number.isFinite(then) || now === 0) return "";
  const seconds = Math.max(0, Math.round((now - then) / 1000));
  if (seconds < 5) return "now";
  if (seconds < 60) return `${seconds}s ago`;
  const minutes = Math.floor(seconds / 60);
  if (minutes < 60) return `${minutes}m ago`;
  const hours = Math.floor(minutes / 60);
  if (hours < 48) return `${hours}h ago`;
  return `${Math.floor(hours / 24)}d ago`;
}

const ABSOLUTE = new Intl.DateTimeFormat("en-GB", {
  dateStyle: "medium",
  timeStyle: "medium",
  timeZone: "UTC",
});

export function absoluteTime(iso: string): string {
  const then = Date.parse(iso);
  return Number.isFinite(then) ? `${ABSOLUTE.format(then)} UTC` : "";
}

export function formatCountdown(seconds: number): string {
  const safe = Math.max(0, Math.floor(seconds));
  const minutes = Math.floor(safe / 60);
  return `${minutes}:${String(safe % 60).padStart(2, "0")}`;
}
