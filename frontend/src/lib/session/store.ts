/**
 * The tab's session and its hold on the seat (PLAN section 16), kept outside React so it can be
 * tested with fake timers and a fake fetch.
 *
 * The rules it follows:
 * - The tab ID lives in memory only, one per page load. `sessionStorage` would be copied by
 *   "Duplicate tab" (F-036), and two tabs would then share one lease.
 * - A heartbeat renews the 45 s lease every 15 s, and says whether the person did anything, which
 *   keeps a guest's idle timer alive.
 * - On `pagehide` the seat is released with `sendBeacon`, so a reload can claim it straight away.
 * - If the lease lapses (a sleeping laptop, or a browser throttling a hidden tab's timers), the
 *   store claims again once, without taking over, so no one else is pushed out by it.
 * - A Web Lock says at once when another tab of this browser has WikiGuard open, before the
 *   server answers. It's only a convenience: the server's lease is the real control.
 *
 * Nothing here decides what anyone may do. The server checks every request.
 */
import {
  ApiError,
  createApiClient,
  isAbort,
  releaseSeat,
  type ApiClient,
} from "../api/client";
import type { Role, SessionInfo } from "../api/types";

export const HEARTBEAT_MS = 15_000;
// One automatic re-claim per window. Without it, a server that keeps answering "seat lost" right
// after a good claim would get re-claims in a tight loop, with no delay between them (F-108).
export const RECLAIM_WINDOW_MS = 30_000;
export const TAB_LOCK = "wikiguard-tab";
const ACTIVITY_EVENTS = ["pointerdown", "keydown", "wheel", "touchstart"] as const;

export type SignedOutReason = "ended" | "expired" | "taken_over" | "signed_out";

export type SessionState =
  /** `since` lets the gate say "waking up" when the API is slow to answer (a sleeping server). */
  | { status: "starting"; since: number }
  | { status: "signed_out"; reason?: SignedOutReason; loginError?: string }
  | { status: "signing_in"; as: Role }
  | {
      status: "seated";
      role: Role;
      csrfToken: string;
      expiresAt: string;
      freshExplainsLeft: number | null;
      /** When a guest's session ends, as epoch ms, or null for `user` and before the first beat. */
      guestEndsAt: number | null;
    }
  | { status: "seat_taken"; retryAfterSeconds: number | null }
  /** The owner is asked before ending someone else's session. */
  | { status: "confirm_take_over"; holderRole: Role | null }
  | { status: "cooldown"; retryAfterSeconds: number }
  /** `byLock`: this browser's other tab was seen by its Web Lock, before asking the server. */
  | { status: "other_tab"; byLock?: boolean }
  /** The lease lapsed again straight after an automatic re-claim. A person decides what next. */
  | { status: "lost" }
  | { status: "unreachable"; error: ApiError };

export type SeatedState = Extract<SessionState, { status: "seated" }>;

export function newTabId(): string {
  // Matches the API's pattern: 8 to 64 of [A-Za-z0-9_-]. A UUID is 36.
  return crypto.randomUUID();
}

/** The part of the Web Locks API this uses, so tests can pass a fake. */
export interface TabLocks {
  request(
    name: string,
    options: { ifAvailable: boolean },
    callback: (lock: unknown) => Promise<void> | void,
  ): Promise<unknown>;
}

export interface SessionStoreOptions {
  fetch?: typeof fetch;
  tabId?: string;
  heartbeatMs?: number;
  release?: typeof releaseSeat;
  /** Defaults to `navigator.locks`. Pass null to switch it off. */
  locks?: TabLocks | null;
}

type Listener = () => void;

function endedReason(error: ApiError): SignedOutReason {
  const reason = error.extra.reason;
  if (reason === "taken_over") return "taken_over";
  if (reason === "signed_out") return "signed_out";
  return error.code === "session_ended" ? "expired" : "ended";
}

function minutes(seconds: number | undefined): string {
  const value = Math.max(1, Math.ceil((seconds ?? 60) / 60));
  return `${value} minute${value === 1 ? "" : "s"}`;
}

/** What a failed owner sign-in says. The API never says which part was wrong beyond this. */
export function loginErrorText(error: ApiError): string | null {
  switch (error.code) {
    case "bad_credentials":
      return "That password isn’t right.";
    case "login_locked":
      return `Too many failed sign-ins. Try again in about ${minutes(error.retryAfterSeconds)}.`;
    case "login_busy":
    case "rate_limited":
      return "Sign-in is busy. Try again in a moment.";
    default:
      return null;
  }
}

export class SessionStore {
  readonly api: ApiClient;
  private tabIdValue: string | null;

  private state: SessionState = { status: "starting", since: 0 };
  private readonly listeners = new Set<Listener>();
  private readonly heartbeatMs: number;
  private readonly release: typeof releaseSeat;
  private readonly locksOption: TabLocks | null | undefined;
  private heartbeatTimer: ReturnType<typeof setTimeout> | null = null;
  private active = false;
  private reclaiming = false;
  private lastReclaimAt = Number.NEGATIVE_INFINITY;
  // The token from /auth/me, sent with the claim that comes before the seat is held.
  private pendingCsrf: string | null = null;
  // Held only between "this will sign the guest out" and the owner's answer, then dropped. The
  // refused sign-in made no session, so taking over means signing in again.
  private pendingPassword: string | null = null;
  // Each start() bumps this, so an answer from an older start can't overwrite a newer one.
  private generation = 0;
  private detach: (() => void) | null = null;
  private releaseLock: (() => void) | null = null;
  private waitingForLock = false;

  constructor(options: SessionStoreOptions = {}) {
    this.tabIdValue = options.tabId ?? null;
    this.heartbeatMs = options.heartbeatMs ?? HEARTBEAT_MS;
    this.release = options.release ?? releaseSeat;
    this.locksOption = options.locks;
    this.api = createApiClient({
      fetch: options.fetch,
      credentials: () => ({
        tabId: this.tabId,
        csrfToken: this.state.status === "seated" ? this.state.csrfToken : this.pendingCsrf,
      }),
      onProblem: (error) => this.onProblem(error),
    });
  }

  /**
   * Made on first use, which is always in the browser. Making it in the constructor would run
   * while the server renders, and every visitor would share that ID.
   */
  get tabId(): string {
    this.tabIdValue ??= newTabId();
    return this.tabIdValue;
  }

  private get locks(): TabLocks | null {
    if (this.locksOption !== undefined) return this.locksOption;
    const native = typeof navigator !== "undefined" ? navigator.locks : undefined;
    return native ? (native as unknown as TabLocks) : null;
  }

  // --- For useSyncExternalStore ---

  subscribe = (listener: Listener): (() => void) => {
    this.listeners.add(listener);
    return () => this.listeners.delete(listener);
  };

  getSnapshot = (): SessionState => this.state;

  private set(next: SessionState): void {
    const wasOtherTab = this.state.status === "other_tab";
    this.state = next;
    if (next.status !== "seated") this.stopHeartbeat();
    // Told by the server that another tab took over: let that tab have the lock.
    if (next.status === "other_tab" && !next.byLock && !wasOtherTab) this.yieldLock();
    for (const listener of this.listeners) listener();
  }

  // --- Page lifecycle ---

  /** Starts listening to the page and resumes any session the cookie still holds. */
  connect(target: Window = window): () => void {
    this.detach?.();
    const markActive = () => {
      this.active = true;
    };
    const onPageHide = () => {
      if (this.state.status === "seated") this.release(this.tabId, this.state.csrfToken);
      this.stopHeartbeat();
    };
    const onPageShow = (event: PageTransitionEvent) => {
      // Restored from the back-forward cache: the seat was given back on pagehide, so claim again.
      if (event.persisted) void this.start();
    };
    for (const name of ACTIVITY_EVENTS) {
      target.addEventListener(name, markActive, { passive: true });
    }
    target.addEventListener("pagehide", onPageHide);
    target.addEventListener("pageshow", onPageShow);
    this.detach = () => {
      for (const name of ACTIVITY_EVENTS) target.removeEventListener(name, markActive);
      target.removeEventListener("pagehide", onPageHide);
      target.removeEventListener("pageshow", onPageShow);
    };
    void this.begin();
    return () => this.disconnect();
  }

  disconnect(): void {
    this.generation += 1;
    this.stopHeartbeat();
    this.detach?.();
    this.detach = null;
    this.releaseLock?.();
    this.releaseLock = null;
  }

  /** Takes this browser's tab lock if it's free, then starts. If not, says so at once. */
  private async begin(): Promise<void> {
    const locks = this.locks;
    if (!locks) return this.start();
    const held = await this.takeLock(locks, true);
    if (held) return this.start();
    this.set({ status: "other_tab", byLock: true });
    this.waitForLock(true);
  }

  /** Resolves true once the lock is held (it stays held until released), false if it's busy. */
  private takeLock(locks: TabLocks, ifAvailable: boolean): Promise<boolean> {
    return new Promise((resolve) => {
      locks
        .request(TAB_LOCK, { ifAvailable }, (lock) => {
          if (!lock) {
            resolve(false);
            return;
          }
          resolve(true);
          return new Promise<void>((released) => {
            this.releaseLock = released;
          });
        })
        // A browser that refuses the request just loses the convenience.
        .catch(() => resolve(true));
    });
  }

  /**
   * Waits for the lock. With `autoStart`, a tab that was only blocked by the lock (never by the
   * server) tries for the seat once it's free, which is when the other tab closed. A tab the server
   * displaced never starts by itself: it would claim, be told "other tab" again, and loop (F-115).
   */
  private waitForLock(autoStart: boolean): void {
    const locks = this.locks;
    if (!locks || this.waitingForLock || this.releaseLock) return;
    this.waitingForLock = true;
    void this.takeLock(locks, false).then(() => {
      this.waitingForLock = false;
      const blocked = this.state.status === "other_tab" && this.state.byLock;
      if (autoStart && blocked) void this.start();
    });
  }

  /** The server says another tab has the seat: let that tab have the lock too, and wait for a person. */
  private yieldLock(): void {
    this.releaseLock?.();
    this.releaseLock = null;
  }

  // --- Actions ---

  /** Reads the session from the cookie, then claims the seat for this tab. */
  async start(): Promise<void> {
    const generation = ++this.generation;
    if (this.state.status !== "seated") this.set({ status: "starting", since: Date.now() });
    let info: SessionInfo;
    try {
      info = await this.api.me();
    } catch (error) {
      if (generation === this.generation) this.fail(error);
      return;
    }
    if (generation !== this.generation) return;
    await this.claim(info, false, generation);
  }

  async signInGuest(): Promise<void> {
    const generation = ++this.generation;
    this.set({ status: "signing_in", as: "guest" });
    try {
      // Signing in as a guest claims the seat in the same request, so there's no race for it.
      const info = await this.api.signInGuest(this.tabId);
      if (generation === this.generation) this.seat(info);
    } catch (error) {
      if (generation === this.generation) this.fail(error);
    }
  }

  /** The owner's sign-in. `takeOver` only after they've confirmed ending the guest's session. */
  async signInUser(password: string, takeOver = false): Promise<void> {
    const generation = ++this.generation;
    this.pendingPassword = null;
    this.set({ status: "signing_in", as: "user" });
    try {
      const info = await this.api.signInUser(password, this.tabId, takeOver);
      if (generation === this.generation) this.seat(info);
    } catch (error) {
      if (generation !== this.generation) return;
      if (error instanceof ApiError && error.code === "confirm_take_over") {
        this.pendingPassword = password;
        this.set({ status: "confirm_take_over", holderRole: holderRole(error) });
        return;
      }
      const message = error instanceof ApiError ? loginErrorText(error) : null;
      if (message) this.set({ status: "signed_out", loginError: message });
      else this.fail(error);
    }
  }

  /** The owner said yes: end the other session and take the seat. */
  async confirmTakeOver(): Promise<void> {
    const password = this.pendingPassword;
    this.pendingPassword = null;
    if (password !== null) return this.signInUser(password, true);
    return this.useHere();
  }

  cancelTakeOver(): void {
    this.showSignIn();
  }

  /** Back to the sign-in screen: from "in use", the owner can still sign in and take over. */
  showSignIn(): void {
    this.pendingPassword = null;
    this.generation += 1;
    this.set({ status: "signed_out" });
  }

  /** Moves the session's seat to this tab from another one (the second-tab screen). */
  async useHere(): Promise<void> {
    const generation = ++this.generation;
    this.set({ status: "starting", since: Date.now() });
    try {
      const info = await this.api.me();
      if (generation === this.generation) await this.claim(info, true, generation);
    } catch (error) {
      if (generation === this.generation) this.fail(error);
    }
  }

  async signOut(): Promise<void> {
    this.generation += 1;
    try {
      await this.api.signOut();
    } catch {
      // Already gone on the server, or the API is down: either way this tab is signed out.
    }
    this.set({ status: "signed_out", reason: "signed_out" });
  }

  /** Re-reads the session, for numbers that change on the server, such as guest explains left. */
  async refresh(): Promise<void> {
    if (this.state.status !== "seated") return;
    try {
      const info = await this.api.me();
      if (this.state.status === "seated") {
        this.set({ ...this.state, freshExplainsLeft: info.fresh_explains_left ?? null });
      }
    } catch {
      // onProblem has already dealt with anything that matters.
    }
  }

  // --- Internals ---

  private async claim(info: SessionInfo, takeOver: boolean, generation: number): Promise<void> {
    this.pendingCsrf = info.csrf_token;
    try {
      await this.api.claim(this.tabId, takeOver);
      if (generation === this.generation) this.seat(info);
    } catch (error) {
      if (generation === this.generation) this.fail(error);
    } finally {
      this.pendingCsrf = null;
    }
  }

  private seatedFrom(info: SessionInfo): SeatedState {
    const previous = this.state.status === "seated" ? this.state : null;
    return {
      status: "seated",
      role: info.role,
      csrfToken: info.csrf_token,
      expiresAt: info.expires_at,
      freshExplainsLeft: info.fresh_explains_left ?? null,
      guestEndsAt: info.role === "guest" ? (previous?.guestEndsAt ?? null) : null,
    };
  }

  private seat(info: SessionInfo): void {
    this.set(this.seatedFrom(info));
    // The tab with the seat holds the lock, so a new tab sees it at once. Quietly: no auto-start.
    this.waitForLock(false);
    // Beat at once, so a guest's timer shows without waiting 15 s. Each beat schedules the next.
    this.stopHeartbeat();
    void this.beat();
  }

  private fail(error: unknown): void {
    if (isAbort(error)) return;
    if (!(error instanceof ApiError)) {
      this.set({
        status: "unreachable",
        error: new ApiError({ status: 0, code: "client_error", title: "Something went wrong" }),
      });
      return;
    }
    switch (error.code) {
      case "not_signed_in":
        this.set({ status: "signed_out" });
        return;
      case "session_ended":
        this.set({ status: "signed_out", reason: endedReason(error) });
        return;
      case "seat_taken":
        this.set({ status: "seat_taken", retryAfterSeconds: error.retryAfterSeconds ?? null });
        return;
      case "confirm_take_over":
        // An owner's session from the cookie, finding someone else seated: ask first.
        this.set({ status: "confirm_take_over", holderRole: holderRole(error) });
        return;
      case "other_tab":
        this.set({ status: "other_tab" });
        return;
      case "guest_cooldown":
        this.set({ status: "cooldown", retryAfterSeconds: error.retryAfterSeconds ?? 60 });
        return;
      default:
        this.set({ status: "unreachable", error });
    }
  }

  /** Problems from any request, including page data. Only seat and session problems matter here. */
  private onProblem(error: ApiError): void {
    if (this.state.status !== "seated") return;
    if (error.code === "seat_lost") {
      if (this.reclaiming) return;
      if (Date.now() - this.lastReclaimAt < RECLAIM_WINDOW_MS) {
        this.set({ status: "lost" });
        return;
      }
      this.lastReclaimAt = Date.now();
      this.reclaiming = true;
      void this.start().finally(() => {
        this.reclaiming = false;
      });
      return;
    }
    if (["not_signed_in", "session_ended", "other_tab"].includes(error.code)) this.fail(error);
  }

  private async beat(): Promise<void> {
    this.heartbeatTimer = null;
    if (this.state.status !== "seated") return;
    const active = this.active;
    this.active = false;
    try {
      const reply = await this.api.heartbeat(active);
      if (this.state.status !== "seated") return;
      const guestEndsAt =
        this.state.role === "guest" ? Date.now() + reply.guest_seconds_left * 1000 : null;
      this.set({ ...this.state, guestEndsAt });
    } catch {
      // Seat and session problems went to onProblem. Anything else (the API briefly down, a rate
      // limit) just means trying again on time: the lease outlasts two missed beats.
    } finally {
      const keepGoing = this.state.status === "seated" && !this.reclaiming;
      if (keepGoing && this.heartbeatTimer === null) this.scheduleHeartbeat();
    }
  }

  private scheduleHeartbeat(): void {
    this.stopHeartbeat();
    this.heartbeatTimer = setTimeout(() => void this.beat(), this.heartbeatMs);
  }

  private stopHeartbeat(): void {
    if (this.heartbeatTimer !== null) clearTimeout(this.heartbeatTimer);
    this.heartbeatTimer = null;
  }
}

function holderRole(error: ApiError): Role | null {
  const role = error.extra.holder_role;
  return role === "guest" || role === "user" ? role : null;
}
