// The Worker's routing and header rules, with Node's own test runner and no packages:
//   node --test edge/worker.test.mjs
// fetch is replaced with a recorder, so nothing leaves the machine.
import assert from "node:assert/strict";
import { afterEach, beforeEach, test } from "node:test";

import worker, { CLIENT_IP_HEADER, EDGE_SECRET_HEADER } from "./worker.js";

const env = {
  API_ORIGIN: "https://wikiguard-api.onrender.com",
  FRONTEND_ORIGIN: "https://wiki-guard.vercel.app",
  EDGE_SHARED_SECRET: "edge-test-secret",
};
const realFetch = globalThis.fetch;
let sent;
let answer;

beforeEach(() => {
  sent = [];
  answer = () => new Response("ok", { status: 200 });
  globalThis.fetch = async (url, init) => {
    sent.push({ url, headers: new Headers(init.headers), method: init.method, redirect: init.redirect });
    return answer(url);
  };
});
afterEach(() => {
  globalThis.fetch = realFetch;
});

function visit(path, headers = {}, method = "GET") {
  return worker.fetch(
    new Request(`https://wikiguard.example.workers.dev${path}`, {
      method,
      headers: { "cf-connecting-ip": "203.0.113.7", ...headers },
      body: method === "POST" ? "{}" : undefined,
    }),
    env,
  );
}

test("/api goes to Render with /api taken off, and the query kept", async () => {
  await visit("/api/v1/queue?limit=5");
  assert.equal(sent[0].url, "https://wikiguard-api.onrender.com/v1/queue?limit=5");
});

test("everything else goes to Vercel unchanged", async () => {
  await visit("/models?x=1");
  assert.equal(sent[0].url, "https://wiki-guard.vercel.app/models?x=1");
  await visit("/apiary");
  assert.equal(sent[1].url, "https://wiki-guard.vercel.app/apiary");
});

test("the API gets the secret and the visitor's address; Vercel gets neither", async () => {
  await visit("/api/v1/health");
  assert.equal(sent[0].headers.get(EDGE_SECRET_HEADER), "edge-test-secret");
  assert.equal(sent[0].headers.get(CLIENT_IP_HEADER), "203.0.113.7");
  assert.ok(sent[0].headers.get("x-request-id"));
  await visit("/");
  assert.equal(sent[1].headers.get(EDGE_SECRET_HEADER), null);
  assert.equal(sent[1].headers.get(CLIENT_IP_HEADER), null);
});

test("a visitor can't fake the secret, an address, or a forwarding chain", async () => {
  await visit("/api/v1/auth/guest", {
    [EDGE_SECRET_HEADER]: "guess",
    [CLIENT_IP_HEADER]: "198.51.100.1",
    "x-forwarded-for": "198.51.100.2",
    "x-real-ip": "198.51.100.3",
    forwarded: "for=198.51.100.4",
    "true-client-ip": "198.51.100.5",
  }, "POST");
  const headers = sent[0].headers;
  assert.equal(headers.get(EDGE_SECRET_HEADER), "edge-test-secret");
  assert.equal(headers.get(CLIENT_IP_HEADER), "203.0.113.7");
  for (const name of ["x-forwarded-for", "x-real-ip", "forwarded", "true-client-ip"]) {
    assert.equal(headers.get(name), null, name);
  }
});

test("cookies and the method pass through, and redirects are never followed here", async () => {
  await visit("/api/v1/auth/guest", { cookie: "wg_session=abc" }, "POST");
  assert.equal(sent[0].headers.get("cookie"), "wg_session=abc");
  assert.equal(sent[0].method, "POST");
  assert.equal(sent[0].redirect, "manual");
});

test("a redirect to an origin is pointed back at the single address", async () => {
  answer = () => new Response(null, { status: 308, headers: { location: "https://wiki-guard.vercel.app/models/" } });
  let response = await visit("/models");
  assert.equal(response.headers.get("location"), "https://wikiguard.example.workers.dev/models/");
  answer = () => new Response(null, { status: 307, headers: { location: "/v1/queue/" } });
  response = await visit("/api/v1/queue");
  assert.equal(response.headers.get("location"), "https://wikiguard.example.workers.dev/api/v1/queue/");
});

test("a redirect to somewhere else is left alone", async () => {
  answer = () => new Response(null, { status: 302, headers: { location: "https://en.wikipedia.org/wiki/Moon" } });
  const response = await visit("/edit/1");
  assert.equal(response.headers.get("location"), "https://en.wikipedia.org/wiki/Moon");
});

test("past the edge limit, /api answers 429 without reaching Render", async () => {
  const limited = { ...env, API_LIMITER: { limit: async ({ key }) => ({ success: key !== "203.0.113.7" }) } };
  const response = await worker.fetch(
    new Request("https://x.workers.dev/api/v1/feed", { headers: { "cf-connecting-ip": "203.0.113.7" } }),
    limited,
  );
  assert.equal(response.status, 429);
  assert.equal(response.headers.get("retry-after"), "60");
  assert.equal((await response.json()).code, "edge_rate_limited");
  assert.equal(sent.length, 0);
});

test("without its secret the Worker refuses /api instead of passing it on", async () => {
  const response = await worker.fetch(
    new Request("https://x.workers.dev/api/v1/feed"),
    { ...env, EDGE_SHARED_SECRET: "" },
  );
  assert.equal(response.status, 503);
  assert.equal(sent.length, 0);
});
