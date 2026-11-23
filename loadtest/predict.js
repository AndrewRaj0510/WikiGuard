// Load test for POST /v1/predict (Phase 8, task 2). Run with scripts/load_test.py.
//
// The owner holds the seat and sends real diffs from the benchmark sample (data/, never committed)
// at a fixed arrival rate, one stage after another. The owner's limit is 60 a minute (LIMIT_PREDICT_
// USER), so the last stage, at twice that, shows the limiter's 429s. Each body gets a unique comment
// suffix, so the API's prediction cache can't answer and every request runs a model, which is
// what a person pasting new text causes. Requests carry the edit's real text, up to the API's
// 20,000-character limit.
import http from "k6/http";
import { check } from "k6";
import { SharedArray } from "k6/data";
import { Counter } from "k6/metrics";
import exec from "k6/execution";
import { API, heartbeat, ownerSession, signOut, summaryTo } from "./common.js";

const rows = new SharedArray("sample", () => JSON.parse(open(__ENV.SAMPLE_PATH)));
const answeredBy = new Counter("predict_answered_by");
const STAGE_SECONDS = Number(__ENV.STAGE_SECONDS || 120);
// Requests a second in each stage: 15, 30, 60, and 120 a minute.
const RATES = [0.25, 0.5, 1, 2];

const scenarios = {
  heartbeat: {
    executor: "constant-arrival-rate",
    rate: 1,
    timeUnit: "15s",
    duration: `${STAGE_SECONDS * RATES.length}s`,
    preAllocatedVUs: 1,
    exec: "beat",
  },
};
const thresholds = {};
RATES.forEach((rate, index) => {
  const name = `predict_${String(rate).replace(".", "_")}`;
  scenarios[name] = {
    executor: "constant-arrival-rate",
    // k6 rates are whole numbers per time unit, so a quarter a second is 1 per 4 s.
    rate: rate >= 1 ? rate : 1,
    timeUnit: rate >= 1 ? "1s" : `${1 / rate}s`,
    duration: `${STAGE_SECONDS}s`,
    startTime: `${STAGE_SECONDS * index}s`,
    preAllocatedVUs: 4,
    maxVUs: 12,
    exec: "predict",
  };
  // Thresholds name the sub-metrics the summary should keep. Nothing aborts on them: the run
  // records, and docs/performance.md compares with the SLO (p95 300 ms, docs/slo.md).
  thresholds[`http_req_duration{scenario:${name},status:200}`] = ["p(95)<300"];
  thresholds[`http_reqs{scenario:${name},status:200}`] = ["count>=0"];
  thresholds[`http_reqs{scenario:${name},status:429}`] = ["count>=0"];
  thresholds[`http_reqs{scenario:${name},status:503}`] = ["count>=0"];
});

// Which model answered, kept in the summary as sub-metrics.
for (const model of ["roberta", "distilbert", "bilstm", "bert", "mlp"]) {
  thresholds[`predict_answered_by{model:${model}}`] = ["count>=0"];
}
thresholds["predict_answered_by{fallback:true}"] = ["count>=0"];

export const options = {
  scenarios,
  thresholds,
  summaryTrendStats: ["min", "med", "avg", "p(90)", "p(95)", "p(99)", "max"],
};

export function setup() {
  return ownerSession("10.0.0.1");
}

export function beat(session) {
  heartbeat(session.headers);
}

export function predict(session) {
  const row = rows[Math.floor(Math.random() * rows.length)];
  const body = {
    added_text: row.added_text.slice(0, 20000),
    removed_text: row.removed_text.slice(0, 20000),
    comment: `${row.comment.slice(0, 900)} #${exec.vu.idInTest}-${exec.vu.iterationInScenario}`,
    is_anonymous: row.is_anonymous,
    is_minor: row.is_minor,
  };
  const reply = http.post(`${API}/v1/predict`, JSON.stringify(body), {
    headers: session.headers,
    tags: { name: "predict" },
    timeout: "30s",
  });
  check(reply, { "answered or refused on purpose": (r) => [200, 429, 503].includes(r.status) });
  if (reply.status === 200) {
    const answer = reply.json();
    answeredBy.add(1, { model: answer.model_name, fallback: String(answer.used_fallback) });
  }
}

export function teardown(session) {
  signOut(session);
}

export function handleSummary(data) {
  return summaryTo(data);
}
