import { afterEach, describe, expect, it } from "vitest";
import type { HealthSummary, ModelEntry } from "./api/types";
import { comparisonOf, healthSignals, retrainRunsOf } from "./health";
import { neighbour } from "./list-order";
import { aliasLabel, chainRank, statsOf } from "./models";
import { contentSecurityPolicy, INDEXABLE_PATHS, newNonce } from "./security/headers";
import { shortcutKey } from "./shortcuts";

describe("the content security policy", () => {
  const csp = contentSecurityPolicy({ nonce: "abc123", dev: false, https: true });

  it("allows only this request's nonce for scripts and styles", () => {
    expect(csp).toContain("script-src 'self' 'nonce-abc123' 'strict-dynamic'");
    expect(csp).toContain("style-src 'self' 'nonce-abc123'");
    expect(csp).not.toContain("unsafe-inline");
    expect(csp).not.toContain("unsafe-eval");
  });

  it("can't be framed, has no plugins, and talks only to its own address", () => {
    expect(csp).toContain("frame-ancestors 'none'");
    expect(csp).toContain("object-src 'none'");
    expect(csp).toContain("base-uri 'none'");
    expect(csp).toContain("connect-src 'self';");
  });

  it("upgrades requests only over https, and loosens only in development", () => {
    expect(csp).toContain("upgrade-insecure-requests");
    expect(contentSecurityPolicy({ nonce: "n", dev: false, https: false })).not.toContain("upgrade");
    const dev = contentSecurityPolicy({ nonce: "n", dev: true, https: false });
    expect(dev).toContain("'unsafe-eval'");
    expect(dev).toContain("connect-src 'self' ws:");
  });

  it("makes a different 128-bit nonce each time", () => {
    const nonces = new Set(Array.from({ length: 50 }, newNonce));
    expect(nonces.size).toBe(50);
    expect(atob([...nonces][0]).length).toBe(16);
  });

  it("lets search engines index only the landing and privacy pages", () => {
    expect([...INDEXABLE_PATHS].sort()).toEqual(["/", "/privacy"]);
  });
});

function summary(overrides: Partial<HealthSummary> = {}): HealthSummary {
  return {
    stale: false,
    scored_last_24h: 1000,
    fallback_rate_24h: 0.01,
    load_shed_rate_24h: 0,
    last_scored_at: "2026-11-09T12:00:00Z",
    unscored_edits: 0,
    models_loaded: true,
    chain: [
      { alias: "champion", model_name: "roberta", model_version: "1", breaker: "closed" },
      { alias: "fallback-4", model_name: "mlp", model_version: "5", breaker: "closed" },
    ],
    live_metrics: [],
    drift_reports: [],
    retrain_runs: [],
    ...overrides,
  };
}

const NOW = Date.parse("2026-11-09T12:01:00Z");

describe("health signals", () => {
  it("says nothing when all is well", () => {
    expect(healthSignals(summary(), NOW)).toEqual([]);
  });

  it("puts an open breaker first, naming the model", () => {
    const chain = summary().chain.map((item, index) =>
      index === 0 ? { ...item, breaker: "open" } : item,
    );
    const [first] = healthSignals(summary({ chain, fallback_rate_24h: 0.3 }), NOW);
    expect(first).toMatchObject({ tone: "danger", title: "Degraded" });
    expect(first.detail).toContain("roberta is failing");
  });

  it("warns when fallbacks answer a tenth or more of a day's edits", () => {
    expect(healthSignals(summary({ fallback_rate_24h: 0.1 }), NOW)[0].detail).toContain("10%");
    expect(healthSignals(summary({ fallback_rate_24h: 0.09 }), NOW)).toEqual([]);
  });

  it("calls scoring stale after ten quiet minutes, but not when nothing was scored at all", () => {
    const late = NOW + 10 * 60_000;
    expect(healthSignals(summary(), late)[0].title).toBe("Stale");
    expect(healthSignals(summary({ scored_last_24h: 0 }), late)).toEqual([]);
  });

  it("ignores breakers when this API process hasn't loaded models", () => {
    const chain = summary().chain.map((item) => ({ ...item, breaker: "open" }));
    expect(healthSignals(summary({ chain, models_loaded: false }), NOW)).toEqual([]);
  });

  it("says when the numbers come from the API's cache", () => {
    expect(healthSignals(summary({ stale: true }), NOW)[0].title).toBe("Showing cached data");
  });
});

describe("before vs after reports", () => {
  it("reads the trainer's comparison", () => {
    expect(
      comparisonOf({
        before: { model: "roberta", version: 1, pr_auc: 0.36, pr_auc_interval: [0.34, 0.39] },
        after: { model: "roberta", version: "6", pr_auc: 0.38 },
        difference: { mean: 0.02, low: 0.005, high: 0.035 },
        promoted: true,
        reason: "clears the gate",
      }),
    ).toEqual({
      before: { model: "roberta", version: "1", prAuc: 0.36, prAucInterval: [0.34, 0.39] },
      after: { model: "roberta", version: "6", prAuc: 0.38, prAucInterval: null },
      difference: { mean: 0.02, low: 0.005, high: 0.035 },
      promoted: true,
      reason: "clears the gate",
    });
  });

  it("is absent for runs without one, and skips malformed rows", () => {
    expect(comparisonOf({})).toBeNull();
    const runs = retrainRunsOf(
      summary({
        retrain_runs: [
          { id: 1, trigger: "manual", status: "done", started_at: "2026-11-09T10:00:00Z", details: {} },
          { trigger: "drift", status: "running" },
        ],
      }),
    );
    expect(runs).toHaveLength(1);
    expect(runs[0]).toMatchObject({ trigger: "manual", comparison: null });
  });
});

function entry(alias: string | null, metrics: Record<string, unknown>): ModelEntry {
  return {
    model_name: "roberta",
    model_version: "1",
    alias,
    feature_version: 1,
    snapshot_hash: "abc",
    metrics,
    registered_at: "2026-11-07T00:00:00Z",
  };
}

describe("model metrics", () => {
  it("reads the synced numbers and card", () => {
    const stats = statsOf(
      entry("champion", {
        pr_auc: 0.3641,
        pr_auc_interval: [0.3385, 0.3872],
        card: {
          slices: { anonymous: { rows: 3508, pr_auc: 0.52 } },
          pr_curve: { points: [[0.01, 1], [1, 0.06]], rows: 25294, rows_scored: 25302 },
        },
      }),
    );
    expect(stats.prAuc).toBe(0.3641);
    expect(stats.prAucInterval).toEqual([0.3385, 0.3872]);
    expect(stats.slices.anonymous).toMatchObject({ rows: 3508, prAuc: 0.52 });
    expect(stats.curve?.points).toHaveLength(2);
    expect(stats.curve?.rows).toBe(25294);
  });

  it("turns anything malformed into 'not measured', never a crash", () => {
    const stats = statsOf(
      entry(null, {
        pr_auc: "0.9",
        pr_auc_interval: [0.5, 0.1],
        card: { slices: { x: "nope" }, pr_curve: { points: "nope" } },
      }),
    );
    expect(stats.prAuc).toBeNull();
    expect(stats.prAucInterval).toBeNull();
    expect(stats.slices).toEqual({});
    expect(stats.curve).toBeNull();
  });

  it("orders the chain and names its aliases", () => {
    const order = ["fallback-2", null, "champion", "fallback-1"].sort(
      (a, b) => chainRank(a) - chainRank(b),
    );
    expect(order).toEqual(["champion", "fallback-1", "fallback-2", null]);
    expect(aliasLabel("fallback-3")).toBe("Fallback 3");
    expect(aliasLabel(null)).toBe("Registered");
  });
});

describe("list order", () => {
  const order = { ids: [5, 4, 3], href: "/queue", label: "Review queue" };
  it("steps forward and back, and stops at the ends", () => {
    expect(neighbour(4, 1, order)).toBe(3);
    expect(neighbour(4, -1, order)).toBe(5);
    expect(neighbour(3, 1, order)).toBeNull();
    expect(neighbour(99, 1, order)).toBeNull();
  });
});

describe("shortcuts", () => {
  afterEach(() => {
    document.body.innerHTML = "";
  });

  function press(key: string, init: KeyboardEventInit = {}, target: EventTarget = document.body) {
    const event = new KeyboardEvent("keydown", { key, ...init });
    Object.defineProperty(event, "target", { value: target });
    return shortcutKey(event);
  }

  it("reads plain keys, and Shift-made ones like ?", () => {
    expect(press("J")).toBe("j");
    expect(press("?", { shiftKey: true })).toBe("?");
  });

  it("stays out of the way of typing, browser shortcuts, and dialogs", () => {
    const input = document.createElement("input");
    expect(press("v", {}, input)).toBeNull();
    expect(press("v", { ctrlKey: true })).toBeNull();
    expect(press("v", { metaKey: true })).toBeNull();
    const dialog = document.createElement("dialog");
    dialog.setAttribute("open", "");
    document.body.append(dialog);
    expect(press("v")).toBeNull();
  });
});
