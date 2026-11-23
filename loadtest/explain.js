// The Explain burst (Phase 8, task 3). Run with scripts/load_test.py, which starts a fake LLM that
// answers after 1.5 s: tests never call a real one (CLAUDE.md).
//
// The owner's normal reads (the feed every 3 s, the heartbeat every 15 s) run the whole time, so the
// summary shows whether a burst of slow LLM calls slows everything else. Three bursts:
//   same_edit   20 requests at once for one edit, from 20 addresses: one LLM call should serve all
//   many_edits  30 requests at once for 30 different edits, from 30 addresses: 30 calls in flight
//   one_address 15 requests from one address in a few seconds: the limit is 10 a minute
import http from "k6/http";
import { check } from "k6";
import { API, asVisitor, heartbeat, ownerSession, signOut, summaryTo } from "./common.js";

const BURSTS = {
  same_edit: { count: 20, start: 30 },
  many_edits: { count: 30, start: 75 },
  one_address: { count: 15, start: 120 },
};

const scenarios = {
  feed: {
    executor: "constant-arrival-rate",
    rate: 1,
    timeUnit: "3s",
    duration: "165s",
    preAllocatedVUs: 2,
    maxVUs: 10,
    exec: "feed",
  },
  heartbeat: {
    executor: "constant-arrival-rate",
    rate: 1,
    timeUnit: "15s",
    duration: "165s",
    preAllocatedVUs: 1,
    exec: "beat",
  },
};
const thresholds = {
  "http_req_duration{scenario:feed}": ["p(95)<200"],
  "http_req_duration{scenario:feed,burst:on}": ["p(95)<200"],
  "http_req_duration{scenario:feed,burst:off}": ["p(95)<200"],
};
for (const [name, burst] of Object.entries(BURSTS)) {
  scenarios[name] = {
    executor: "per-vu-iterations",
    vus: burst.count,
    iterations: 1,
    startTime: `${burst.start}s`,
    maxDuration: "40s",
    exec: "explain",
    env: { BURST: name },
  };
  thresholds[`http_req_duration{scenario:${name}}`] = ["p(95)<60000"];
  for (const status of [200, 429, 503]) {
    thresholds[`http_reqs{scenario:${name},status:${status}}`] = ["count>=0"];
  }
}

export const options = {
  scenarios,
  thresholds,
  summaryTrendStats: ["min", "med", "avg", "p(90)", "p(95)", "p(99)", "max"],
};

export function setup() {
  const session = ownerSession("10.0.0.1");
  const items = http.get(`${API}/v1/queue?limit=50`, { headers: session.headers }).json("items");
  return { session, revIds: items.map((item) => item.rev_id), started: Date.now() };
}

// Whether a burst is in flight: from its start to 20 s after, by the clock of the setup.
function inBurst(state) {
  const seconds = (Date.now() - state.started) / 1000;
  return Object.values(BURSTS).some((b) => seconds >= b.start && seconds < b.start + 20);
}

export function feed(state) {
  http.get(`${API}/v1/feed?limit=50`, {
    headers: state.session.headers,
    tags: { name: "feed", burst: inBurst(state) ? "on" : "off" },
  });
}

export function beat(state) {
  heartbeat(state.session.headers);
}

export function explain(state) {
  const burst = __ENV.BURST;
  const vu = __VU;
  let rev = state.revIds[0];
  let headers = asVisitor(state.session, 100 + vu);
  if (burst === "many_edits") rev = state.revIds[1 + (vu % (state.revIds.length - 1))];
  if (burst === "one_address") {
    rev = state.revIds[vu % state.revIds.length];
    headers = asVisitor(state.session, 99);
  }
  const reply = http.post(`${API}/v1/explain/${rev}`, "{}", {
    headers,
    tags: { name: "explain" },
    timeout: "60s",
  });
  check(reply, { "answered or refused on purpose": (r) => [200, 429, 503].includes(r.status) });
}

export function teardown(state) {
  signOut(state.session);
}

export function handleSummary(data) {
  return summaryTo(data);
}
