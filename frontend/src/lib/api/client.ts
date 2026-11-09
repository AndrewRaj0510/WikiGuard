/**
 * The one way the browser talks to the API.
 *
 * - Same-origin `/api/...` only, so the session cookie is first-party (PLAN section 16).
 * - Every request carries the tab's ID, and writes carry the CSRF token. The cookie is `HttpOnly`,
 *   so script never sees it: the token is what proves a write came from this page.
 * - Errors arrive as RFC 9457 problems and become `ApiError`, with the API's own code.
 * - GETs use the browser's normal HTTP cache. The API sends an `ETag` with `no-cache`, so the
 *   browser revalidates every poll, and an unchanged one costs a 304 with no body.
 */
import type { paths } from "./schema";
import type {
  AdminRequestInfo,
  AdminRequestList,
  Decision,
  EditDetail,
  EditorFilter,
  EditPage,
  ExplainReply,
  GateStatus,
  HealthSummary,
  HeartbeatReply,
  ModelList,
  ReviewReply,
  SessionInfo,
} from "./types";

export const API_BASE = "/api";

type Query = Record<string, string | number | boolean | null | undefined>;
type Method = "GET" | "POST" | "PUT";

export interface ProblemInit {
  status: number;
  code: string;
  title: string;
  detail?: string;
  retryAfterSeconds?: number;
  requestId?: string;
  extra?: Record<string, unknown>;
}

/** A problem from the API, or a stand-in when the API couldn't be reached (status 0). */
export class ApiError extends Error {
  readonly status: number;
  readonly code: string;
  readonly detail?: string;
  readonly retryAfterSeconds?: number;
  readonly requestId?: string;
  readonly extra: Record<string, unknown>;

  constructor(init: ProblemInit) {
    super(init.title);
    this.name = "ApiError";
    this.status = init.status;
    this.code = init.code;
    this.detail = init.detail;
    this.retryAfterSeconds = init.retryAfterSeconds;
    this.requestId = init.requestId;
    this.extra = init.extra ?? {};
  }

  /** True when no answer came from the API itself: it's down, asleep, or the network failed. */
  get unreachable(): boolean {
    return this.code === "api_unreachable";
  }
}

export function isAbort(error: unknown): boolean {
  return error instanceof DOMException && error.name === "AbortError";
}

/**
 * Fills a path template from the generated schema. Typing `template` as a schema key means that a
 * route renamed in the API breaks the build here, not at runtime.
 */
export function fillPath(
  template: keyof paths,
  params: Record<string, string | number> = {},
): string {
  return template.replace(/\{(\w+)\}/g, (_, name: string) => {
    const value = params[name];
    if (value === undefined) throw new Error(`Missing path parameter: ${name}`);
    return encodeURIComponent(String(value));
  });
}

function withQuery(path: string, query?: Query): string {
  if (!query) return path;
  const search = new URLSearchParams();
  for (const [key, value] of Object.entries(query)) {
    if (value !== undefined && value !== null) search.set(key, String(value));
  }
  const text = search.toString();
  return text ? `${path}?${text}` : path;
}

function retryAfter(response: Response, body: Record<string, unknown> | undefined) {
  const header = Number(response.headers.get("retry-after"));
  if (Number.isFinite(header) && header > 0) return header;
  const fromBody = body?.retry_after_seconds;
  return typeof fromBody === "number" && fromBody > 0 ? fromBody : undefined;
}

async function readJson(response: Response): Promise<Record<string, unknown> | undefined> {
  const type = response.headers.get("content-type") ?? "";
  if (!type.includes("json")) return undefined;
  try {
    return (await response.json()) as Record<string, unknown>;
  } catch {
    return undefined;
  }
}

function toApiError(response: Response, body: Record<string, unknown> | undefined): ApiError {
  const isProblem = typeof body?.code === "string" && typeof body?.title === "string";
  if (!isProblem) {
    // No problem body: the request never reached the API (a proxy or the host answered instead).
    return new ApiError({
      status: response.status,
      code: response.status >= 500 ? "api_unreachable" : "unexpected_response",
      title:
        response.status >= 500
          ? "WikiGuard's API isn't answering"
          : "WikiGuard's API sent an unexpected answer",
      retryAfterSeconds: retryAfter(response, body),
    });
  }
  const { code, title, detail, request_id, status, type, ...extra } = body as Record<
    string,
    unknown
  >;
  void status;
  void type;
  return new ApiError({
    status: response.status,
    code: code as string,
    title: title as string,
    detail: typeof detail === "string" ? detail : undefined,
    requestId: typeof request_id === "string" ? request_id : undefined,
    retryAfterSeconds: retryAfter(response, body),
    extra,
  });
}

export interface Credentials {
  tabId: string | null;
  csrfToken: string | null;
}

export interface ClientOptions {
  credentials: () => Credentials;
  /** Told about every problem, so the session can react to a lost seat wherever it shows up. */
  onProblem?: (error: ApiError) => void;
  fetch?: typeof fetch;
  base?: string;
}

export interface RequestOptions {
  query?: Query;
  body?: unknown;
  signal?: AbortSignal;
}

export function createApiClient(options: ClientOptions) {
  const base = options.base ?? API_BASE;
  const doFetch = options.fetch ?? ((...args: Parameters<typeof fetch>) => fetch(...args));

  async function request<T>(method: Method, path: string, init: RequestOptions = {}): Promise<T> {
    const headers: Record<string, string> = { accept: "application/json" };
    const { tabId, csrfToken } = options.credentials();
    if (tabId) headers["x-tab-id"] = tabId;
    if (method !== "GET" && csrfToken) headers["x-csrf-token"] = csrfToken;
    if (init.body !== undefined) headers["content-type"] = "application/json";

    let response: Response;
    try {
      response = await doFetch(`${base}${withQuery(path, init.query)}`, {
        method,
        headers,
        body: init.body === undefined ? undefined : JSON.stringify(init.body),
        signal: init.signal,
        credentials: "same-origin",
      });
    } catch (cause) {
      if (isAbort(cause)) throw cause;
      throw new ApiError({
        status: 0,
        code: "api_unreachable",
        title: "WikiGuard's API can't be reached",
      });
    }

    if (response.status === 204) return undefined as T;
    const body = await readJson(response);
    if (!response.ok) {
      const error = toApiError(response, body);
      options.onProblem?.(error);
      throw error;
    }
    if (body === undefined) {
      throw new ApiError({
        status: response.status,
        code: "unexpected_response",
        title: "WikiGuard's API sent an unexpected answer",
      });
    }
    return body as T;
  }

  return {
    request,

    me: (signal?: AbortSignal) =>
      request<SessionInfo>("GET", fillPath("/v1/auth/me"), { signal }),
    signInGuest: (tabId: string) =>
      request<SessionInfo>("POST", fillPath("/v1/auth/guest"), { body: { tab_id: tabId } }),
    signOut: () => request<void>("POST", fillPath("/v1/auth/logout")),

    gateStatus: (signal?: AbortSignal) =>
      request<GateStatus>("GET", fillPath("/v1/gate/status"), { signal }),
    claim: (tabId: string, takeOver = false) =>
      request<void>("POST", fillPath("/v1/gate/claim"), {
        body: { tab_id: tabId, take_over: takeOver },
      }),
    heartbeat: (active: boolean) =>
      request<HeartbeatReply>("POST", fillPath("/v1/gate/heartbeat"), { body: { active } }),

    feed: (params: { limit?: number; cursor?: string | null }, signal?: AbortSignal) =>
      request<EditPage>("GET", fillPath("/v1/feed"), { query: params, signal }),
    queue: (
      params: { editor?: EditorFilter; limit?: number; cursor?: string | null },
      signal?: AbortSignal,
    ) => request<EditPage>("GET", fillPath("/v1/queue"), { query: params, signal }),
    edit: (revId: number, signal?: AbortSignal) =>
      request<EditDetail>("GET", fillPath("/v1/edits/{rev_id}", { rev_id: revId }), { signal }),
    explain: (revId: number) =>
      request<ExplainReply>("POST", fillPath("/v1/explain/{rev_id}", { rev_id: revId })),

    signInUser: (password: string, tabId: string, takeOver = false) =>
      request<SessionInfo>("POST", fillPath("/v1/auth/login"), {
        body: { password, tab_id: tabId, take_over: takeOver },
      }),
    review: (revId: number, decision: Decision, note?: string) =>
      request<ReviewReply>("PUT", fillPath("/v1/edits/{rev_id}/review", { rev_id: revId }), {
        body: note ? { decision, note } : { decision },
      }),
    models: (signal?: AbortSignal) =>
      request<ModelList>("GET", fillPath("/v1/models"), { signal }),
    health: (signal?: AbortSignal) =>
      request<HealthSummary>("GET", fillPath("/v1/health/summary"), { signal }),
    /** Outside /v1 and the schema: true once the API has its database and models. No session. */
    ready: (signal?: AbortSignal) => request<{ ready: boolean }>("GET", "/ready", { signal }),

    // Admin routes: hidden from the public schema, so these paths aren't checked against it.
    adminRequests: (signal?: AbortSignal) =>
      request<AdminRequestList>("GET", "/v1/admin/requests", { signal }),
    requestRetrain: (reason: string) =>
      request<AdminRequestInfo>("POST", "/v1/admin/retrain", { body: { reason } }),
    requestPromote: (modelVersion: string, reason: string) =>
      request<AdminRequestInfo>("POST", "/v1/admin/promote", {
        body: { model_version: modelVersion, reason },
      }),
    requestRollback: (reason: string) =>
      request<AdminRequestInfo>("POST", "/v1/admin/rollback", { body: { reason } }),
  };
}

export type ApiClient = ReturnType<typeof createApiClient>;

/**
 * Gives the seat back as the page goes away, so a reload doesn't wait out the 45 s lease.
 * `sendBeacon` survives unload but can't set headers, so the CSRF token travels in the body.
 */
export function releaseSeat(tabId: string, csrfToken: string, base = API_BASE): boolean {
  if (typeof navigator === "undefined" || !navigator.sendBeacon) return false;
  const body = new Blob([JSON.stringify({ tab_id: tabId, csrf_token: csrfToken })], {
    type: "application/json",
  });
  return navigator.sendBeacon(`${base}${fillPath("/v1/gate/release")}`, body);
}
