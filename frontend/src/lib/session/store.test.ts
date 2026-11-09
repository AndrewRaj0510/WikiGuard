import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { SessionStore } from "./store";
import { createFakeApi, json, problem, SESSION } from "@/test/fake-api";

const TAB = "tab-aaaaaaaa";
const BEAT = json({ guest_seconds_left: 600, lease_seconds: 45 });

function setup() {
  const api = createFakeApi();
  const release = vi.fn(() => true);
  const store = new SessionStore({ fetch: api.fetch, tabId: TAB, release });
  return { api, store, release };
}

/** Lets pending promises (fetch answers) run without moving the clock. */
async function settle() {
  await vi.advanceTimersByTimeAsync(0);
}

beforeEach(() => {
  vi.useFakeTimers();
  vi.setSystemTime(new Date("2026-11-08T12:00:00Z"));
});
afterEach(() => vi.useRealTimers());

describe("starting", () => {
  it("is signed out when the cookie holds no session", async () => {
    const { api, store } = setup();
    api.on("GET", "/v1/auth/me", problem(401, "not_signed_in"));
    store.connect(window);
    await settle();
    expect(store.getSnapshot()).toEqual({ status: "signed_out" });
    store.disconnect();
  });

  it("resumes a session: claims the seat with the CSRF token, then beats at once", async () => {
    const { api, store } = setup();
    api.on("GET", "/v1/auth/me", json(SESSION));
    api.on("POST", "/v1/gate/claim", { status: 204 });
    api.on("POST", "/v1/gate/heartbeat", BEAT);
    store.connect(window);
    await settle();

    const [claim] = api.calls("POST", "/v1/gate/claim");
    expect(claim.body).toEqual({ tab_id: TAB, take_over: false });
    expect(claim.headers["x-csrf-token"]).toBe("csrf-abc");
    expect(api.calls("POST", "/v1/gate/heartbeat")).toHaveLength(1);
    expect(store.getSnapshot()).toMatchObject({
      status: "seated",
      role: "guest",
      freshExplainsLeft: 5,
      guestEndsAt: Date.now() + 600_000,
    });
    store.disconnect();
  });

  it("reports another tab holding the session's seat, and can move it here", async () => {
    const { api, store } = setup();
    api.on("GET", "/v1/auth/me", json(SESSION));
    api.on("POST", "/v1/gate/claim", (request) =>
      (request.body as { take_over: boolean }).take_over
        ? { status: 204 }
        : problem(409, "other_tab"),
    );
    api.on("POST", "/v1/gate/heartbeat", BEAT);
    store.connect(window);
    await settle();
    expect(store.getSnapshot().status).toBe("other_tab");

    await store.useHere();
    await settle();
    expect(store.getSnapshot().status).toBe("seated");
    store.disconnect();
  });

  it("says when the seat is taken, and roughly for how long", async () => {
    const { api, store } = setup();
    api.on("GET", "/v1/auth/me", problem(401, "not_signed_in"));
    api.on("POST", "/v1/auth/guest", problem(409, "seat_taken", { retry_after_seconds: 240 }));
    store.connect(window);
    await settle();
    await store.signInGuest();
    expect(store.getSnapshot()).toEqual({ status: "seat_taken", retryAfterSeconds: 240 });
    store.disconnect();
  });

  it("explains why a session ended", async () => {
    const { api, store } = setup();
    api.on("GET", "/v1/auth/me", problem(401, "session_ended", { reason: "taken_over" }));
    store.connect(window);
    await settle();
    expect(store.getSnapshot()).toEqual({ status: "signed_out", reason: "taken_over" });
    store.disconnect();
  });
});

describe("holding the seat", () => {
  async function seated() {
    const context = setup();
    context.api.on("GET", "/v1/auth/me", problem(401, "not_signed_in"));
    context.api.on("POST", "/v1/auth/guest", json(SESSION));
    context.api.on("POST", "/v1/gate/heartbeat", BEAT);
    context.store.connect(window);
    await settle();
    await context.store.signInGuest();
    await settle();
    return context;
  }

  it("signs in as a guest with this tab's ID", async () => {
    const { api, store } = await seated();
    expect(api.calls("POST", "/v1/auth/guest")[0].body).toEqual({ tab_id: TAB });
    expect(store.getSnapshot().status).toBe("seated");
    store.disconnect();
  });

  it("beats every 15 s, and reports activity only when there was some", async () => {
    const { api, store } = await seated();
    const beats = () => api.calls("POST", "/v1/gate/heartbeat").map((r) => r.body);
    expect(beats()).toEqual([{ active: false }]);

    await vi.advanceTimersByTimeAsync(15_000);
    expect(beats()).toHaveLength(2);
    expect(beats()[1]).toEqual({ active: false });

    window.dispatchEvent(new KeyboardEvent("keydown"));
    await vi.advanceTimersByTimeAsync(15_000);
    expect(beats()[2]).toEqual({ active: true });
    store.disconnect();
  });

  it("keeps beating after a failed beat, so a blip doesn't lose the lease", async () => {
    const { api, store } = await seated();
    api.on("POST", "/v1/gate/heartbeat", problem(429, "rate_limited"));
    await vi.advanceTimersByTimeAsync(15_000);
    api.on("POST", "/v1/gate/heartbeat", BEAT);
    await vi.advanceTimersByTimeAsync(15_000);
    expect(api.calls("POST", "/v1/gate/heartbeat")).toHaveLength(3);
    expect(store.getSnapshot().status).toBe("seated");
    store.disconnect();
  });

  it("claims again, without taking over, when the lease lapsed", async () => {
    const { api, store } = await seated();
    api.on("GET", "/v1/auth/me", json(SESSION));
    api.on("POST", "/v1/gate/claim", { status: 204 });
    let lapses = 1;
    api.on("POST", "/v1/gate/heartbeat", () => (lapses-- > 0 ? problem(409, "seat_lost") : BEAT));
    await vi.advanceTimersByTimeAsync(15_000);
    expect(api.calls("POST", "/v1/gate/claim")[0].body).toEqual({ tab_id: TAB, take_over: false });
    expect(store.getSnapshot().status).toBe("seated");
    store.disconnect();
  });

  it("stops re-claiming if the lease keeps lapsing, instead of looping (F-108)", async () => {
    const { api, store } = await seated();
    api.on("GET", "/v1/auth/me", json(SESSION));
    api.on("POST", "/v1/gate/claim", { status: 204 });
    api.on("POST", "/v1/gate/heartbeat", problem(409, "seat_lost"));
    await vi.advanceTimersByTimeAsync(15_000);
    expect(api.calls("POST", "/v1/gate/claim")).toHaveLength(1);
    expect(store.getSnapshot()).toEqual({ status: "lost" });
    store.disconnect();
  });

  it("gives the seat back on pagehide, with the token in the body", async () => {
    const { store, release } = await seated();
    window.dispatchEvent(new Event("pagehide"));
    expect(release).toHaveBeenCalledWith(TAB, "csrf-abc");
    store.disconnect();
  });

  it("stops beating once signed out", async () => {
    const { api, store } = await seated();
    api.on("POST", "/v1/auth/logout", { status: 204 });
    await store.signOut();
    expect(store.getSnapshot()).toEqual({ status: "signed_out", reason: "signed_out" });
    const before = api.calls("POST", "/v1/gate/heartbeat").length;
    await vi.advanceTimersByTimeAsync(60_000);
    expect(api.calls("POST", "/v1/gate/heartbeat")).toHaveLength(before);
    store.disconnect();
  });

  it("moves to the second-tab screen when a page request finds another tab", async () => {
    const { api, store } = await seated();
    api.on("GET", "/v1/feed", problem(409, "other_tab"));
    await store.api.feed({}).catch(() => undefined);
    expect(store.getSnapshot().status).toBe("other_tab");
    store.disconnect();
  });
});

describe("the tab ID", () => {
  it("is made only when first used, and fits the API's pattern", () => {
    const random = vi.spyOn(crypto, "randomUUID");
    const store = new SessionStore();
    expect(random).not.toHaveBeenCalled();
    expect(store.tabId).toMatch(/^[A-Za-z0-9_-]{8,64}$/);
    expect(store.tabId).toBe(store.tabId);
  });
});
