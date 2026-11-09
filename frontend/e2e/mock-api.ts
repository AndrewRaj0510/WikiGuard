/**
 * A fake WikiGuard API inside the browser, through Playwright's request interception. Tests run
 * against the real production build with no backend, no Docker, and nothing that could reach
 * Wikimedia or an LLM (CLAUDE.md). It enforces roles and CSRF like the real API, so "a guest can't
 * review" is tested, not assumed. The real API's own tests cover the server side.
 */
import type { Page, Route } from "@playwright/test";

export const OWNER_PASSWORD = "owner-e2e-password";
const CSRF = "csrf-e2e-token";

export interface MockEdit {
  rev_id: number;
  title: string;
  comment: string | null;
  added_text: string | null;
  removed_text: string | null;
  score: number;
  is_anonymous: boolean;
  minutes_ago: number;
  bytes_delta?: number;
}

const SAFE_EDITS: MockEdit[] = [
  { rev_id: 101, title: "Ada Lovelace", comment: "fix a date", added_text: "born 1815", removed_text: "born 1816", score: 0.62, is_anonymous: true, minutes_ago: 3 },
  { rev_id: 102, title: "Tidal locking", comment: null, added_text: "lorem ipsum", removed_text: "", score: 0.31, is_anonymous: false, minutes_ago: 5 },
  { rev_id: 103, title: "Basalt", comment: "copyedit", added_text: "a fine-grained rock", removed_text: "a rock", score: 0.08, is_anonymous: false, minutes_ago: 8 },
  { rev_id: 104, title: "Kelp forest", comment: "typo", added_text: "kelp", removed_text: "kelpp", score: 0.04, is_anonymous: true, minutes_ago: 13 },
];

export interface MockOptions {
  edits?: MockEdit[];
  explanation?: string;
}

export class MockApi {
  role: "guest" | "user" | null = null;
  down = false;
  freshExplains = 5;
  reviews = new Map<number, string>();
  calls: { method: string; path: string; body: unknown; headers: Record<string, string> }[] = [];
  edits: MockEdit[];
  explanation: string;

  constructor(options: MockOptions = {}) {
    this.edits = options.edits ?? SAFE_EDITS;
    this.explanation = options.explanation ?? "Changed the birth year from 1816 to 1815.";
  }

  async install(page: Page): Promise<void> {
    await page.route("**/api/**", (route) => this.handle(route));
  }

  /** Adds an edit, as if it just arrived from the stream. */
  arrive(edit: MockEdit): void {
    this.edits = [edit, ...this.edits];
  }

  private summary(edit: MockEdit) {
    const editedAt = new Date(Date.now() - edit.minutes_ago * 60_000).toISOString();
    return {
      rev_id: edit.rev_id,
      title: edit.title,
      edited_at: editedAt,
      is_minor: false,
      is_anonymous: edit.is_anonymous,
      comment: edit.comment,
      bytes_delta: edit.bytes_delta ?? (edit.added_text?.length ?? 0) - (edit.removed_text?.length ?? 0),
      score: edit.score,
      model_name: "roberta",
      model_version: "1",
      threshold: 0.9777,
      used_fallback: false,
      load_shed: false,
      decision: this.reviews.get(edit.rev_id) ?? null,
      content_hidden: false,
      // The real API builds this from the integer; the app never uses it for links anyway.
      wikipedia_url: `https://en.wikipedia.org/w/index.php?diff=${edit.rev_id}`,
    };
  }

  private session() {
    return {
      role: this.role,
      csrf_token: CSRF,
      expires_at: new Date(Date.now() + 15 * 60_000).toISOString(),
      fresh_explains_left: this.role === "guest" ? this.freshExplains : null,
    };
  }

  private async handle(route: Route): Promise<void> {
    if (this.down) return route.abort("connectionrefused");
    const request = route.request();
    const url = new URL(request.url());
    const path = url.pathname.replace(/^\/api/, "");
    const method = request.method();
    let body: unknown;
    try {
      body = request.postDataJSON();
    } catch {
      body = undefined;
    }
    const headers = request.headers();
    this.calls.push({ method, path, body, headers });

    const json = (status: number, data: unknown) =>
      route.fulfill({ status, contentType: "application/json", body: JSON.stringify(data) });
    const problem = (status: number, code: string, extra: Record<string, unknown> = {}) =>
      route.fulfill({
        status,
        contentType: "application/problem+json",
        body: JSON.stringify({ type: `/problems/${code}`, title: code, status, code, request_id: "e2e", ...extra }),
      });
    const writeAllowed = () => headers["x-csrf-token"] === CSRF;

    if (path === "/ready") return json(200, { ready: true, model_loaded: true, database: true });
    if (path === "/v1/gate/status") {
      return json(200, { enabled: true, state: this.role ? "in_use" : "free", retry_after_seconds: null, mine: false });
    }
    if (path === "/v1/auth/me") return this.role ? json(200, this.session()) : problem(401, "not_signed_in");
    if (path === "/v1/auth/guest" && method === "POST") {
      this.role = "guest";
      return json(200, this.session());
    }
    if (path === "/v1/auth/login" && method === "POST") {
      if ((body as { password?: string })?.password !== OWNER_PASSWORD) return problem(401, "bad_credentials");
      this.role = "user";
      return json(200, this.session());
    }
    if (!this.role) return problem(401, "not_signed_in");
    if (path === "/v1/auth/logout") {
      this.role = null;
      return route.fulfill({ status: 204 });
    }
    if (path === "/v1/gate/claim" || path === "/v1/gate/release") return route.fulfill({ status: 204 });
    if (path === "/v1/gate/heartbeat") {
      return json(200, { guest_seconds_left: this.role === "guest" ? 600 : 0, lease_seconds: 45 });
    }

    if (path === "/v1/queue") {
      const editor = url.searchParams.get("editor") ?? "all";
      const items = this.edits
        .filter((e) => !this.reviews.has(e.rev_id))
        .filter((e) => editor === "all" || e.is_anonymous === (editor === "logged_out"))
        .sort((a, b) => b.score - a.score)
        .map((e) => this.summary(e));
      return json(200, { items, next_cursor: null, stale: false });
    }
    if (path === "/v1/feed") {
      const items = [...this.edits].sort((a, b) => a.minutes_ago - b.minutes_ago).map((e) => this.summary(e));
      return json(200, { items, next_cursor: null, stale: false });
    }
    const review = /^\/v1\/edits\/(\d+)\/review$/.exec(path);
    if (review && method === "PUT") {
      if (!writeAllowed()) return problem(403, "bad_csrf");
      if (this.role !== "user") return problem(403, "forbidden");
      const decision = (body as { decision: string }).decision;
      this.reviews.set(Number(review[1]), decision);
      return json(200, { rev_id: Number(review[1]), decision, updated: false });
    }
    const detail = /^\/v1\/edits\/(\d+)$/.exec(path);
    if (detail) {
      const edit = this.edits.find((e) => e.rev_id === Number(detail[1]));
      if (!edit) return problem(404, "edit_not_found");
      return json(200, {
        ...this.summary(edit),
        parent_rev_id: edit.rev_id - 1,
        added_text: edit.added_text,
        removed_text: edit.removed_text,
        comment_hidden: false,
        user_hidden: false,
        user_edit_count: edit.is_anonymous ? null : 120,
        scores: [
          {
            model_name: "roberta",
            model_version: "1",
            alias: "champion",
            score: edit.score,
            threshold: 0.9777,
            used_fallback: false,
            load_shed: false,
            forced: false,
            latency_ms: 31,
            scored_at: new Date().toISOString(),
          },
        ],
        label: null,
        label_excluded_reason: null,
        has_explanation: false,
      });
    }
    const explain = /^\/v1\/explain\/(\d+)$/.exec(path);
    if (explain && method === "POST") {
      if (!writeAllowed()) return problem(403, "bad_csrf");
      if (this.role === "guest") this.freshExplains = Math.max(0, this.freshExplains - 1);
      return json(200, {
        rev_id: Number(explain[1]),
        explanation: this.explanation,
        available: true,
        cached: false,
        reason: null,
        llm_model: "mock-model",
      });
    }
    if (path === "/v1/models") return json(200, { items: MODELS, stale: false });
    if (path === "/v1/health/summary") return json(200, HEALTH);
    if (path.startsWith("/v1/admin/")) {
      if (this.role !== "user") return problem(403, "forbidden");
      if (method === "GET") return json(200, { items: [] });
      return json(202, {
        id: 1,
        kind: path.split("/").pop(),
        status: "pending",
        reason: (body as { reason: string }).reason,
        requested_by: "user",
        requested_at: new Date().toISOString(),
        picked_up_at: null,
        finished_at: null,
        details: {},
      });
    }
    return problem(404, "not_found");
  }
}

const curve = (top: number) =>
  Array.from({ length: 100 }, (_, i) => [(i + 1) / 100, Math.max(0.06, top - i * (top / 110))]);

/** Test data shaped like model_versions, not real results: only screenshots show real numbers. */
export const MODELS = [
  { name: "roberta", version: "1", alias: "champion", pr: 0.36 },
  { name: "mlp", version: "5", alias: "fallback-1", pr: 0.27 },
].map((m) => ({
  model_name: m.name,
  model_version: m.version,
  alias: m.alias,
  feature_version: 1,
  snapshot_hash: "0".repeat(64),
  registered_at: "2026-11-07T00:00:00Z",
  metrics: {
    pr_auc: m.pr,
    pr_auc_interval: [m.pr - 0.02, m.pr + 0.02],
    recall_at_90_precision: 0.05,
    recall_at_90_precision_interval: [0.03, 0.07],
    precision_at_50: 0.9,
    roc_auc: 0.8,
    ece_after: 0.01,
    served_bytes: 1e8,
    card: {
      threshold_rule: "90_percent_precision",
      at_threshold: { precision: 0.95, recall: 0.03, f1: 0.06 },
      slices: { anonymous: { rows: 100, positives: 20, pr_auc: 0.5, false_positive_rate: 0.001 } },
      pr_curve: { points: curve(1), rows: 1000, rows_scored: 1000, prevalence: 0.06 },
    },
  },
}));

export const HEALTH = {
  stale: false,
  scored_last_24h: 1200,
  fallback_rate_24h: 0.01,
  load_shed_rate_24h: 0,
  last_scored_at: new Date().toISOString(),
  unscored_edits: 0,
  models_loaded: true,
  chain: [
    { alias: "champion", model_name: "roberta", model_version: "1", breaker: "closed" },
    { alias: "fallback-1", model_name: "mlp", model_version: "5", breaker: "closed" },
  ],
  live_metrics: [
    { day: "2026-11-07", model_version: "1", metrics: { pr_auc: 0.34 } },
    { day: "2026-11-08", model_version: "1", metrics: { pr_auc: 0.35 } },
  ],
  drift_reports: [{ created_at: "2026-11-08T03:00:00Z", drifted: false }],
  retrain_runs: [
    {
      id: 1,
      trigger: "scheduled",
      status: "done",
      started_at: "2026-11-08T04:00:00Z",
      finished_at: "2026-11-08T06:00:00Z",
      details: {
        before: { model: "roberta", version: "1", pr_auc: 0.36, pr_auc_interval: [0.34, 0.38] },
        after: { model: "roberta", version: "6", pr_auc: 0.37, pr_auc_interval: [0.35, 0.39] },
        difference: { mean: 0.01, low: -0.004, high: 0.024 },
        promoted: false,
        reason: "The interval includes zero.",
      },
    },
  ],
};
