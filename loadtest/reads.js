// Load test for the read endpoints (Phase 8, task 3): normal traffic, a 10x spike, and the soak.
// Run with scripts/load_test.py, which sets MODE (normal, spike, or soak) and DURATION.
//
// "Normal" is what the app sees with its one seat taken (PLAN section 16), at the frontend's own
// polling intervals:
//   - the seated owner, busy: the feed every 3 s, the queue every 10 s, an edit opened every 10 s,
//     the health summary every 60 s (the status banner), the model list every 5 minutes, and the
//     gate heartbeat every 15 s. About 0.63 requests a second.
//   - 3 visitors waiting for the seat, each checking /v1/gate/status every 30 s (the landing page
//     and the "in use" screen), and one guest sign-in attempt a minute, refused while the seat is held.
// "Spike" multiplies every rate by 10. The seated tab's ten copies each come from their own
// address, so the per-IP read limit (120 a minute) applies to each as it would to ten people.
//
// PACING=fixed (the default) fires each flow on a fixed clock, so flows with related intervals
// land on the same instants forever: a worst case for a 0.1 CPU limit, where requests in the same
// 100 ms share 10 ms of CPU. PACING=browser does what the frontend's usePoll does: each flow asks,
// waits for the answer, then waits its interval, so one tab's polls start together and drift
// apart; the spike's other copies start at a random point in their interval, as other people would.
import http from "k6/http";
import { check, sleep } from "k6";
import { API, addressOf, asVisitor, edgeHeaders, heartbeat, ownerSession, signOut, summaryTo } from "./common.js";

const MODE = __ENV.MODE || "normal";
const FACTOR = MODE === "spike" ? 10 : 1;
const DURATION = __ENV.DURATION || "5m";

// [name, endpoint, one request every N seconds, seated?]
const FLOWS = [
  ["feed", "/v1/feed?limit=50", 3, true],
  ["queue", "/v1/queue?limit=50", 10, true],
  ["edit", null, 10, true],
  ["health", "/v1/health/summary", 60, true],
  ["models", "/v1/models", 300, true],
  ["heartbeat", null, 15, true],
  ["gate_status", "/v1/gate/status", 10, false], // 3 visitors, one every 30 s each
  ["guest_sign_in", null, 60, false],
];

const PACING = __ENV.PACING || "fixed";
const scenarios = {};
const thresholds = {};
for (const [name, , every] of FLOWS) {
  scenarios[name] =
    PACING === "browser"
      ? {
          executor: "constant-vus",
          vus: FACTOR,
          duration: DURATION,
          exec: "loop",
          env: { FLOW: name, EVERY: String(every) },
        }
      : {
          executor: "constant-arrival-rate",
          rate: FACTOR,
          timeUnit: `${every}s`,
          duration: DURATION,
          preAllocatedVUs: Math.max(2, FACTOR),
          maxVUs: 20 * FACTOR,
          exec: "flow",
          env: { FLOW: name },
        };
  thresholds[`http_req_duration{scenario:${name}}`] = ["p(95)<200"];
  thresholds[`http_reqs{scenario:${name}}`] = ["count>=0"];
}
thresholds["http_req_failed"] = ["rate<1"];
thresholds["http_reqs{status:429}"] = ["count>=0"];
thresholds["http_reqs{status:409}"] = ["count>=0"];
thresholds["http_reqs{status:503}"] = ["count>=0"];

export const options = {
  scenarios,
  thresholds,
  summaryTrendStats: ["min", "med", "avg", "p(90)", "p(95)", "p(99)", "max"],
};

export function setup() {
  const session = ownerSession("10.0.0.1");
  const queue = http.get(`${API}/v1/queue?limit=50`, { headers: session.headers }).json("items");
  return { session, revIds: queue.map((item) => item.rev_id) };
}

export function flow(state) {
  const name = __ENV.FLOW;
  const vu = __VU;
  // The seated tab's requests, from one of FACTOR addresses.
  const seated = asVisitor(state.session, 1 + (vu % FACTOR));
  let reply;
  if (name === "heartbeat") {
    reply = heartbeat(seated);
  } else if (name === "edit") {
    const rev = state.revIds[Math.floor(Math.random() * state.revIds.length)];
    reply = http.get(`${API}/v1/edits/${rev}`, { headers: seated, tags: { name: "edit" } });
  } else if (name === "guest_sign_in") {
    // Visitors' addresses start past the seated ones, so their limits are their own.
    reply = http.post(`${API}/v1/auth/guest`, JSON.stringify({ tab_id: `visitor-${vu}-${Date.now()}` }), {
      headers: edgeHeaders(addressOf(1000 + vu)),
      tags: { name: "guest_sign_in" },
    });
  } else {
    const headers = FLOWS.find((f) => f[0] === name)[3] ? seated : edgeHeaders(addressOf(1000 + vu));
    reply = http.get(`${API}${FLOWS.find((f) => f[0] === name)[1]}`, { headers, tags: { name } });
  }
  const expected = name === "guest_sign_in" ? [409, 429] : [200, 204];
  check(reply, { "expected status": (r) => expected.includes(r.status) });
}

// PACING=browser: one virtual user per copy, asking, then waiting its interval, like usePoll.
export function loop(state) {
  const every = Number(__ENV.EVERY);
  if (__ITER === 0 && FACTOR > 1 && __VU % FACTOR !== 1) sleep(Math.random() * every);
  flow(state);
  sleep(every);
}

export function teardown(state) {
  signOut(state.session);
}

export function handleSummary(data) {
  return summaryTo(data);
}
