import { describe, expect, it } from "vitest";
import { formatBytesDelta, formatCountdown, relativeTime } from "./format";
import { ELEVATED_SCORE, formatScore, riskBand } from "./risk";
import { articleUrl, diffUrl, isWikipediaUrl, parseRevId } from "./wikipedia";
import { backoffDelay } from "./use-poll";
import { mergePages, parseEditor } from "@/components/views/queue-view";
import { newArrivals } from "@/components/views/feed-view";
import type { EditPage, EditSummary } from "./api/types";

describe("risk bands", () => {
  it("flags at the model's own threshold, before the elevated edge", () => {
    expect(riskBand(0.95, 0.9)).toBe("flagged");
    expect(riskBand(0.9, 0.9)).toBe("flagged");
    expect(riskBand(0.89, 0.9)).toBe("elevated");
  });

  it("is elevated from 0.18 (DEC-061) and low below it", () => {
    expect(ELEVATED_SCORE).toBe(0.18);
    expect(riskBand(0.18, 0.9)).toBe("elevated");
    expect(riskBand(0.1799, 0.9)).toBe("low");
  });

  it("works without a threshold, and with no score at all", () => {
    expect(riskBand(0.5, null)).toBe("elevated");
    expect(riskBand(null, 0.9)).toBe("unscored");
    expect(riskBand(Number.NaN, 0.9)).toBe("unscored");
  });

  it("formats scores as percentages without claiming certainty", () => {
    expect(formatScore(0.4249)).toBe("42%");
    expect(formatScore(0.004)).toBe("<1%");
    expect(formatScore(0.996)).toBe(">99%");
    expect(formatScore(0)).toBe("0%");
    expect(formatScore(null)).toBe("—");
  });
});

describe("formatting", () => {
  it("signs byte changes with a real minus", () => {
    expect(formatBytesDelta(1234)).toBe("+1,234");
    expect(formatBytesDelta(-56)).toBe("−56");
    expect(formatBytesDelta(0)).toBe("0");
    expect(formatBytesDelta(null)).toBe("—");
  });

  it("gives short relative times", () => {
    const now = Date.parse("2026-11-08T12:00:00Z");
    expect(relativeTime("2026-11-08T11:59:58Z", now)).toBe("now");
    expect(relativeTime("2026-11-08T11:59:18Z", now)).toBe("42s ago");
    expect(relativeTime("2026-11-08T11:53:00Z", now)).toBe("7m ago");
    expect(relativeTime("2026-11-06T00:00:00Z", now)).toBe("2d ago");
    expect(relativeTime("2026-11-08T11:00:00Z", 0)).toBe("");
  });

  it("counts down in minutes and seconds", () => {
    expect(formatCountdown(605.4)).toBe("10:05");
    expect(formatCountdown(-3)).toBe("0:00");
  });
});

describe("Wikipedia links", () => {
  it("builds diff links from integers only", () => {
    expect(diffUrl(1234)).toBe("https://en.wikipedia.org/w/index.php?diff=1234");
    expect(() => diffUrl(1.5)).toThrow();
    expect(() => diffUrl(-1)).toThrow();
  });

  it("encodes titles, so one can't steer a link elsewhere", () => {
    const url = articleUrl('AC/DC"><script>&action=delete');
    expect(isWikipediaUrl(url)).toBe(true);
    expect(url).not.toContain("<");
    expect(new URL(url).searchParams.get("action")).toBeNull();
    expect(new URL(url).searchParams.get("title")).toBe('AC/DC"><script>&action=delete');
  });

  it("recognises only https en.wikipedia.org", () => {
    expect(isWikipediaUrl("https://en.wikipedia.org/wiki/X")).toBe(true);
    expect(isWikipediaUrl("http://en.wikipedia.org/wiki/X")).toBe(false);
    expect(isWikipediaUrl("https://en.wikipedia.org.evil.test/")).toBe(false);
    expect(isWikipediaUrl("javascript:alert(1)")).toBe(false);
  });

  it("accepts only plain positive integers as rev IDs", () => {
    expect(parseRevId("1234")).toBe(1234);
    expect(parseRevId("0")).toBeNull();
    expect(parseRevId("12a")).toBeNull();
    expect(parseRevId("1e5")).toBeNull();
    expect(parseRevId("99999999999999999")).toBeNull();
    expect(parseRevId(undefined)).toBeNull();
  });
});

describe("polling", () => {
  it("backs off by doubling up to a minute, and honours Retry-After", () => {
    expect(backoffDelay(3000, 0)).toBe(3000);
    expect(backoffDelay(3000, 1)).toBe(6000);
    expect(backoffDelay(3000, 3)).toBe(24000);
    expect(backoffDelay(3000, 10)).toBe(60000);
    expect(backoffDelay(3000, 1, 30)).toBe(30000);
  });
});

function row(rev_id: number): EditSummary {
  return {
    rev_id,
    title: `T${rev_id}`,
    edited_at: "2026-11-08T12:00:00Z",
    is_minor: false,
    is_anonymous: false,
    comment: null,
    bytes_delta: 1,
    score: 0.5,
    model_name: "roberta",
    model_version: "1",
    threshold: 0.9,
    used_fallback: false,
    load_shed: false,
    decision: null,
    content_hidden: false,
    wikipedia_url: "",
  };
}

function page(ids: number[]): EditPage {
  return { items: ids.map(row), next_cursor: null, stale: false };
}

describe("lists", () => {
  it("merges queue pages, keeping each edit once, first page first", () => {
    const merged = mergePages(page([5, 4, 3]), [page([3, 2]), page([2, 1])]);
    expect(merged.map((item) => item.rev_id)).toEqual([5, 4, 3, 2, 1]);
  });

  it("reads the editor filter from the URL, defaulting to all", () => {
    expect(parseEditor("logged_out")).toBe("logged_out");
    expect(parseEditor("registered")).toBe("registered");
    expect(parseEditor("<script>")).toBe("all");
    expect(parseEditor(null)).toBe("all");
  });

  it("marks only rows that weren't there before, and none on the first load", () => {
    expect(newArrivals(null, page([3, 2, 1]))).toEqual(new Set());
    expect(newArrivals(new Set([2, 1]), page([4, 3, 2]))).toEqual(new Set([4, 3]));
  });
});
