import { describe, expect, it, vi } from "vitest";
import { ApiError, createApiClient, fillPath } from "./client";
import { createFakeApi, json, problem } from "@/test/fake-api";

function client(api = createFakeApi(), csrfToken: string | null = "csrf-1") {
  const onProblem = vi.fn();
  return {
    api,
    onProblem,
    client: createApiClient({
      fetch: api.fetch,
      credentials: () => ({ tabId: "tab-12345678", csrfToken }),
      onProblem,
    }),
  };
}

describe("requests", () => {
  it("sends the tab ID on reads but never the CSRF token", async () => {
    const { api, client: c } = client();
    api.on("GET", "/v1/feed", json({ items: [], next_cursor: null, stale: false }));
    await c.feed({ limit: 50 });
    const [sent] = api.calls("GET", "/v1/feed");
    expect(sent.headers["x-tab-id"]).toBe("tab-12345678");
    expect(sent.headers["x-csrf-token"]).toBeUndefined();
    expect(sent.path).toBe("/v1/feed?limit=50");
  });

  it("sends the CSRF token and a JSON body on writes", async () => {
    const { api, client: c } = client();
    api.on("POST", "/v1/gate/heartbeat", json({ guest_seconds_left: 10, lease_seconds: 45 }));
    await c.heartbeat(true);
    const [sent] = api.calls("POST", "/v1/gate/heartbeat");
    expect(sent.headers["x-csrf-token"]).toBe("csrf-1");
    expect(sent.headers["content-type"]).toBe("application/json");
    expect(sent.body).toEqual({ active: true });
  });

  it("leaves out empty query values", async () => {
    const { api, client: c } = client();
    api.on("GET", "/v1/queue", json({ items: [], next_cursor: null, stale: false }));
    await c.queue({ editor: "logged_out", cursor: null, limit: 50 });
    expect(api.calls("GET", "/v1/queue")[0].path).toBe("/v1/queue?editor=logged_out&limit=50");
  });
});

describe("errors", () => {
  it("turns a problem into an ApiError with its code, retry time, and extras", async () => {
    const { api, client: c, onProblem } = client();
    api.on("POST", "/v1/auth/guest", {
      ...problem(409, "seat_taken", { retry_after_seconds: 300 }),
    });
    const error = await c.signInGuest("tab-12345678").catch((e: unknown) => e);
    expect(error).toBeInstanceOf(ApiError);
    expect(error).toMatchObject({
      status: 409,
      code: "seat_taken",
      retryAfterSeconds: 300,
      requestId: "req-1",
    });
    expect((error as ApiError).extra).toEqual({ retry_after_seconds: 300 });
    expect(onProblem).toHaveBeenCalledWith(error);
  });

  it("prefers the Retry-After header", async () => {
    const { api, client: c } = client();
    const answer = problem(429, "rate_limited");
    api.on("GET", "/v1/feed", { ...answer, headers: { ...answer.headers, "retry-after": "7" } });
    const error = (await c.feed({}).catch((e: unknown) => e)) as ApiError;
    expect(error.retryAfterSeconds).toBe(7);
  });

  it("calls an HTML 502 from a proxy unreachable, not an API answer", async () => {
    const { api, client: c } = client();
    api.on("GET", "/v1/feed", {
      status: 502,
      body: "<html>Bad gateway</html>",
      headers: { "content-type": "text/html" },
    });
    const error = (await c.feed({}).catch((e: unknown) => e)) as ApiError;
    expect(error.code).toBe("api_unreachable");
    expect(error.unreachable).toBe(true);
  });

  it("calls a network failure unreachable", async () => {
    const failing = (() => Promise.reject(new TypeError("Failed to fetch"))) as typeof fetch;
    const c = createApiClient({ fetch: failing, credentials: () => ({ tabId: null, csrfToken: null }) });
    const error = (await c.me().catch((e: unknown) => e)) as ApiError;
    expect(error.unreachable).toBe(true);
    expect(error.status).toBe(0);
  });

  it("lets an abort through untouched, so callers can ignore it", async () => {
    const aborting = (() =>
      Promise.reject(new DOMException("aborted", "AbortError"))) as typeof fetch;
    const c = createApiClient({ fetch: aborting, credentials: () => ({ tabId: null, csrfToken: null }) });
    await expect(c.me()).rejects.toMatchObject({ name: "AbortError" });
  });
});

describe("fillPath", () => {
  it("fills and encodes path parameters", () => {
    expect(fillPath("/v1/edits/{rev_id}", { rev_id: 42 })).toBe("/v1/edits/42");
    expect(fillPath("/v1/edits/{rev_id}", { rev_id: "a/b?c" })).toBe("/v1/edits/a%2Fb%3Fc");
  });

  it("refuses a missing parameter", () => {
    expect(() => fillPath("/v1/edits/{rev_id}")).toThrow(/rev_id/);
  });
});
