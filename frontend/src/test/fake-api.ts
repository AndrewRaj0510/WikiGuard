/**
 * A stand-in for the API: routes by method and path, records every request, and answers with
 * whatever each test sets. Tests never touch a real API or the network.
 */
export interface Recorded {
  method: string;
  path: string;
  headers: Record<string, string>;
  body: unknown;
}

type Answer = { status: number; body?: unknown; headers?: Record<string, string> };
type Handler = (request: Recorded) => Answer;

export function problem(status: number, code: string, extra: Record<string, unknown> = {}): Answer {
  return {
    status,
    body: { type: `/problems/${code}`, title: code, status, code, request_id: "req-1", ...extra },
    headers: { "content-type": "application/problem+json" },
  };
}

export function json(body: unknown, status = 200): Answer {
  return { status, body, headers: { "content-type": "application/json" } };
}

export function createFakeApi() {
  const routes = new Map<string, Handler>();
  const requests: Recorded[] = [];

  const fetch = async (input: RequestInfo | URL, init: RequestInit = {}): Promise<Response> => {
    const url = new URL(String(input), "http://localhost");
    const method = (init.method ?? "GET").toUpperCase();
    const path = url.pathname.replace(/^\/api/, "");
    const recorded: Recorded = {
      method,
      path: path + url.search,
      headers: { ...(init.headers as Record<string, string>) },
      body: typeof init.body === "string" ? JSON.parse(init.body) : undefined,
    };
    requests.push(recorded);
    const handler = routes.get(`${method} ${path}`);
    const answer = handler ? handler(recorded) : problem(404, "not_found");
    if (answer.status === 204) return new Response(null, { status: 204 });
    return new Response(answer.body === undefined ? null : JSON.stringify(answer.body), {
      status: answer.status,
      headers: answer.headers,
    });
  };

  return {
    fetch: fetch as typeof globalThis.fetch,
    requests,
    on(method: string, path: string, handler: Handler | Answer) {
      routes.set(`${method} ${path}`, typeof handler === "function" ? handler : () => handler);
    },
    calls(method: string, path: string) {
      return requests.filter((r) => r.method === method && r.path.split("?")[0] === path);
    },
  };
}

export const SESSION = {
  role: "guest",
  csrf_token: "csrf-abc",
  expires_at: "2026-11-08T12:15:00Z",
  fresh_explains_left: 5,
} as const;
