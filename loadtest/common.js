// Shared by the k6 scripts in this folder (Phase 8 load tests; run them with scripts/load_test.py).
//
// The app has one seat (PLAN section 16): one signed-in tab at a time. So the owner signs in once,
// in setup(), and every virtual user that plays that tab reuses its cookie, tab id, and CSRF token.
// k6's setup() runs apart from the virtual users and doesn't share their cookie jars, which is why
// the session travels as a plain Cookie header.
//
// Each visitor gets its own address in x-wikiguard-client-ip, signed with the edge secret, as the
// Cloudflare Worker sends it in production. Without that, every request would come from 127.0.0.1
// and share one set of per-IP limits, which no real visitor does.
import http from "k6/http";

export const API = __ENV.API || "http://127.0.0.1:8766";
const EDGE_SECRET = __ENV.EDGE_SECRET;

/** A distinct private address per visitor number. */
export function addressOf(visitor) {
  return `10.${(visitor >> 16) & 255}.${(visitor >> 8) & 255}.${visitor & 255}`;
}

export function edgeHeaders(address) {
  return {
    "content-type": "application/json",
    "x-edge-secret": EDGE_SECRET,
    "x-wikiguard-client-ip": address,
  };
}

/** Sign in as the owner, taking the seat. Returns what each virtual user needs to act as that tab. */
export function ownerSession(address) {
  const tab = `loadtest-${Date.now()}`;
  const reply = http.post(
    `${API}/v1/auth/login`,
    JSON.stringify({ password: __ENV.OWNER_PASSWORD, tab_id: tab, take_over: true }),
    { headers: edgeHeaders(address), tags: { name: "login" } },
  );
  if (reply.status !== 200) throw new Error(`owner sign-in failed: ${reply.status} ${reply.body}`);
  const cookie = reply.cookies.wikiguard_session[0].value;
  return {
    address,
    headers: {
      ...edgeHeaders(address),
      "x-tab-id": tab,
      "x-csrf-token": reply.json("csrf_token"),
      cookie: `wikiguard_session=${cookie}`,
    },
  };
}

/** The same session from another address: for the spike, where the seated tab's load is multiplied. */
export function asVisitor(session, visitor) {
  return { ...session.headers, "x-wikiguard-client-ip": addressOf(visitor) };
}

export function heartbeat(headers) {
  return http.post(`${API}/v1/gate/heartbeat`, JSON.stringify({ active: true }), {
    headers,
    tags: { name: "heartbeat" },
  });
}

export function signOut(session) {
  http.post(`${API}/v1/auth/logout`, "{}", { headers: session.headers, tags: { name: "logout" } });
}

/** A k6 summary trimmed to what docs/performance.md needs, written where the runner asked. */
export function summaryTo(data) {
  const out = __ENV.OUT || "summary.json";
  return { [out]: JSON.stringify(data, null, 2), stdout: `\nSummary written to ${out}\n` };
}
