// WikiGuard's single site address (D5, DEC-016, F-035): a Cloudflare Worker on workers.dev.
//
//   /api/...  ->  the API on Render, with /api taken off the front (DEC-063)
//   anything  ->  the frontend on Vercel
//
// One address means the session cookie is first-party, so browsers that block third-party cookies
// still keep the user signed in. The Worker also:
// - adds the shared secret, so the API can refuse anything that didn't come through here
// - passes the visitor's address in its own header, which only the secret makes believable
// - removes every forwarding header a visitor could send, so none can be spoofed through
// - limits /api requests per visitor at the edge, before they wake Render
// - rewrites redirects from either origin back to this address
//
// At Cloudflare's free daily limit (100,000 requests) requests fail with error 1027. Nothing is
// billed (DEC-001). Settings: API_ORIGIN and FRONTEND_ORIGIN in wrangler.toml, and the secret
// EDGE_SHARED_SECRET from `wrangler secret put`, never in a file.

export const EDGE_SECRET_HEADER = "x-edge-secret";
export const CLIENT_IP_HEADER = "x-wikiguard-client-ip";
const REQUEST_ID_HEADER = "x-request-id";
// Anything a visitor could use to claim another address or to fake the Worker.
const SPOOFABLE = [
  "x-forwarded-for",
  "x-forwarded-host",
  "x-forwarded-proto",
  "x-real-ip",
  "forwarded",
  "true-client-ip",
  EDGE_SECRET_HEADER,
  CLIENT_IP_HEADER,
];

export default {
  async fetch(request, env) {
    const url = new URL(request.url);
    const toApi = url.pathname === "/api" || url.pathname.startsWith("/api/");
    if (toApi && env.API_LIMITER) {
      const visitor = request.headers.get("cf-connecting-ip") || "unknown";
      const { success } = await env.API_LIMITER.limit({ key: visitor });
      if (!success) {
        return problem(429, "edge_rate_limited", "Too many requests. Try again in a minute.", {
          "retry-after": "60",
        });
      }
    }
    if (toApi && !env.EDGE_SHARED_SECRET) {
      // Fail closed: without the secret the API would refuse everything anyway.
      return problem(503, "edge_not_configured", "The site isn't fully set up yet.");
    }
    const origin = toApi ? env.API_ORIGIN : env.FRONTEND_ORIGIN;
    const target = new URL(origin);
    target.pathname = toApi ? url.pathname.slice("/api".length) || "/" : url.pathname;
    target.search = url.search;

    const headers = new Headers(request.headers);
    for (const name of SPOOFABLE) headers.delete(name);
    headers.set("x-forwarded-host", url.host);
    headers.set("x-forwarded-proto", "https");
    if (toApi) {
      headers.set(EDGE_SECRET_HEADER, env.EDGE_SHARED_SECRET);
      headers.set(CLIENT_IP_HEADER, request.headers.get("cf-connecting-ip") || "");
      if (!headers.has(REQUEST_ID_HEADER)) headers.set(REQUEST_ID_HEADER, crypto.randomUUID());
    }

    const response = await fetch(target.toString(), {
      method: request.method,
      headers,
      body: ["GET", "HEAD"].includes(request.method) ? undefined : request.body,
      // A redirect goes back to the browser, rewritten to this address, never followed here.
      redirect: "manual",
    });
    return rewriteRedirect(response, target.origin, url.origin, toApi);
  },
};

// A Location pointing at an origin is pointed back at the Worker, so the browser never leaves the
// single address and never sees Render's or Vercel's own name.
export function rewriteRedirect(response, originBase, publicBase, toApi) {
  const location = response.headers.get("location");
  if (!location) return response;
  const resolved = new URL(location, originBase);
  if (resolved.origin !== new URL(originBase).origin) return response;
  const prefix = toApi ? "/api" : "";
  const rewritten = new URL(`${prefix}${resolved.pathname}${resolved.search}`, publicBase);
  const copy = new Response(response.body, response);
  copy.headers.set("location", rewritten.toString());
  return copy;
}

function problem(status, code, detail, extra = {}) {
  return new Response(JSON.stringify({ type: "about:blank", title: detail, status, code, detail }), {
    status,
    headers: { "content-type": "application/problem+json", "cache-control": "no-store", ...extra },
  });
}
