import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { SessionStore, TAB_LOCK, type TabLocks } from "./store";
import { createFakeApi, json, problem, SESSION } from "@/test/fake-api";

const TAB = "tab-bbbbbbbb";
const OWNER = { ...SESSION, role: "user", fresh_explains_left: null } as const;
const BEAT = json({ guest_seconds_left: 0, lease_seconds: 45 });

beforeEach(() => vi.useFakeTimers());
afterEach(() => vi.useRealTimers());

async function settle() {
  await vi.advanceTimersByTimeAsync(0);
}

function setup(locks: TabLocks | null = null) {
  const api = createFakeApi();
  api.on("GET", "/v1/auth/me", problem(401, "not_signed_in"));
  api.on("POST", "/v1/gate/heartbeat", BEAT);
  const store = new SessionStore({ fetch: api.fetch, tabId: TAB, release: () => true, locks });
  return { api, store };
}

describe("the owner's sign-in", () => {
  it("signs in with the password and this tab's ID", async () => {
    const { api, store } = setup();
    api.on("POST", "/v1/auth/login", json(OWNER));
    store.connect(window);
    await settle();
    await store.signInUser("correct horse");
    await settle();
    expect(api.calls("POST", "/v1/auth/login")[0].body).toEqual({
      password: "correct horse",
      tab_id: TAB,
      take_over: false,
    });
    expect(store.getSnapshot()).toMatchObject({ status: "seated", role: "user", guestEndsAt: null });
    store.disconnect();
  });

  it("says when the password is wrong, without saying more", async () => {
    const { api, store } = setup();
    api.on("POST", "/v1/auth/login", problem(401, "bad_credentials"));
    store.connect(window);
    await settle();
    await store.signInUser("nope");
    expect(store.getSnapshot()).toEqual({
      status: "signed_out",
      loginError: "That password isn’t right.",
    });
    store.disconnect();
  });

  it("says how long a lockout lasts", async () => {
    const { api, store } = setup();
    const locked = problem(429, "login_locked");
    api.on("POST", "/v1/auth/login", { ...locked, headers: { ...locked.headers, "retry-after": "840" } });
    store.connect(window);
    await settle();
    await store.signInUser("nope");
    expect(store.getSnapshot()).toMatchObject({
      status: "signed_out",
      loginError: "Too many failed sign-ins. Try again in about 14 minutes.",
    });
    store.disconnect();
  });

  it("asks before taking the seat from a guest, then sends take_over", async () => {
    const { api, store } = setup();
    api.on("POST", "/v1/auth/login", (request) =>
      (request.body as { take_over: boolean }).take_over
        ? json(OWNER)
        : problem(409, "confirm_take_over", { holder_role: "guest" }),
    );
    store.connect(window);
    await settle();
    await store.signInUser("correct horse");
    expect(store.getSnapshot()).toEqual({ status: "confirm_take_over", holderRole: "guest" });

    await store.confirmTakeOver();
    await settle();
    const [first, second] = api.calls("POST", "/v1/auth/login").map((r) => r.body);
    expect(first).toMatchObject({ take_over: false });
    expect(second).toEqual({ password: "correct horse", tab_id: TAB, take_over: true });
    expect(store.getSnapshot().status).toBe("seated");
    store.disconnect();
  });

  it("forgets the password when the owner cancels", async () => {
    const { api, store } = setup();
    api.on("POST", "/v1/auth/login", problem(409, "confirm_take_over", { holder_role: "guest" }));
    store.connect(window);
    await settle();
    await store.signInUser("correct horse");
    store.cancelTakeOver();
    expect(store.getSnapshot()).toEqual({ status: "signed_out" });
    // With nothing remembered, confirming now can only move this session's own seat.
    api.on("GET", "/v1/auth/me", problem(401, "not_signed_in"));
    await store.confirmTakeOver();
    expect(api.calls("POST", "/v1/auth/login")).toHaveLength(1);
    store.disconnect();
  });
});

/** A Web Locks stand-in: one holder at a time, and a queue of waiters. */
function fakeLocks() {
  let held = false;
  const waiters: (() => void)[] = [];
  const locks: TabLocks = {
    request(name, options, callback) {
      expect(name).toBe(TAB_LOCK);
      if (held && options.ifAvailable) return Promise.resolve(callback(null));
      const grant = () => {
        held = true;
        return Promise.resolve(callback({ name })).then(() => {
          held = false;
          waiters.shift()?.();
        });
      };
      if (!held) return grant();
      return new Promise((resolve) => waiters.push(() => resolve(grant())));
    },
  };
  return {
    locks,
    holdElsewhere() {
      let release = () => {};
      void locks.request(TAB_LOCK, { ifAvailable: false }, () => new Promise<void>((r) => (release = r)));
      return () => release();
    },
  };
}

describe("the tab lock", () => {
  it("says at once that another tab has WikiGuard open, without asking the server", async () => {
    const fake = fakeLocks();
    const releaseOther = fake.holdElsewhere();
    const { api, store } = setup(fake.locks);
    store.connect(window);
    await settle();
    expect(store.getSnapshot()).toEqual({ status: "other_tab", byLock: true });
    expect(api.requests).toHaveLength(0);

    // The other tab closes: this one gets the lock and goes on to check its session.
    releaseOther();
    await settle();
    expect(api.calls("GET", "/v1/auth/me")).toHaveLength(1);
    expect(store.getSnapshot()).toEqual({ status: "signed_out" });
    store.disconnect();
  });

  it("starts normally when no other tab holds the lock", async () => {
    const fake = fakeLocks();
    const { api, store } = setup(fake.locks);
    store.connect(window);
    await settle();
    expect(api.calls("GET", "/v1/auth/me")).toHaveLength(1);
    store.disconnect();
  });

  it("lets go of the lock when the server says another tab took over", async () => {
    const fake = fakeLocks();
    const { api, store } = setup(fake.locks);
    api.on("POST", "/v1/auth/guest", json(SESSION));
    store.connect(window);
    await settle();
    await store.signInGuest();
    await settle();
    api.on("GET", "/v1/feed", problem(409, "other_tab"));
    await store.api.feed({}).catch(() => undefined);
    expect(store.getSnapshot()).toEqual({ status: "other_tab" });
    await settle();
    // Released, and not taken back: another tab asking now gets it at once, and this tab
    // waits for a person instead of claiming again in a loop (F-115).
    expect(api.calls("GET", "/v1/auth/me")).toHaveLength(1);
    let granted = false;
    await fake.locks.request(TAB_LOCK, { ifAvailable: true }, (lock) => {
      granted = lock !== null;
    });
    expect(granted).toBe(true);
    store.disconnect();
  });
});
