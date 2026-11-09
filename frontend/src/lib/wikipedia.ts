/**
 * Links out go only to English Wikipedia, and are built here from an integer or an encoded
 * title. Nothing an editor typed can choose where a link points (PLAN section 8).
 */
export const WIKIPEDIA_ORIGIN = "https://en.wikipedia.org";
/** For every external link: no window.opener, no referrer, and no endorsement for search engines. */
export const EXTERNAL_REL = "noopener noreferrer nofollow";

export function diffUrl(revId: number): string {
  if (!Number.isSafeInteger(revId) || revId <= 0) throw new Error("Not a revision ID");
  return `${WIKIPEDIA_ORIGIN}/w/index.php?diff=${revId}`;
}

export function articleUrl(title: string): string {
  return `${WIKIPEDIA_ORIGIN}/w/index.php?title=${encodeURIComponent(title)}`;
}

/** True only for an https link to en.wikipedia.org, for any URL that came from elsewhere. */
export function isWikipediaUrl(url: string): boolean {
  try {
    return new URL(url).origin === WIKIPEDIA_ORIGIN;
  } catch {
    return false;
  }
}

/** Parses a route's rev ID: digits only, and a safe integer, or null. */
export function parseRevId(raw: string | string[] | undefined): number | null {
  if (typeof raw !== "string" || !/^[1-9]\d{0,15}$/.test(raw)) return null;
  const value = Number(raw);
  return Number.isSafeInteger(value) ? value : null;
}
