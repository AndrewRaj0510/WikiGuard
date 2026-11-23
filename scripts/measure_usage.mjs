/**
 * What one visitor costs the free tiers, page by page (scripts/measure_usage.py, Phase 8).
 *
 * Opens one page through the real Worker in Chromium (scripts/edge_local.mjs), as a guest or not
 * signed in, stays on it for --seconds, and counts every request the browser sent, split the way
 * the free limits count them:
 *   - worker: every request, since everything goes through the Worker (100,000 a day);
 *   - api: requests to /api, which reach Render;
 *   - vercel_functions: documents and RSC payloads, which Vercel renders per request (DEC-065);
 *   - static: /_next/static files, served from Vercel's cache.
 * Bytes are the response bodies' sizes. The first --load-seconds are the page load; the rest is
 * the steady polling, reported per minute.
 *
 *   node scripts/measure_usage.mjs --seconds 120 --edit 123
 * The last line is "RESULT {json}".
 */
import { createRequire } from "node:module";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { parseArgs } from "node:util";

const root = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const require = createRequire(join(root, "frontend", "package.json"));
const { chromium } = require("@playwright/test");

const { values: args } = parseArgs({
  options: {
    base: { type: "string", default: "http://localhost:8787" },
    // The landing page, not signed in, then each of these as a guest.
    paths: { type: "string", default: "/queue,/feed,/models,/health" },
    edit: { type: "string" },
    seconds: { type: "string", default: "120" },
    "load-seconds": { type: "string", default: "10" },
  },
});
const seconds = Number(args.seconds);
const loadSeconds = Number(args["load-seconds"]);

function kind(url) {
  const { pathname, search } = new URL(url);
  if (pathname.startsWith("/api/")) return "api";
  if (pathname.startsWith("/_next/static/")) return "static";
  if (search.includes("_rsc=") || !pathname.includes(".")) return "vercel_functions";
  return "static";
}

function sum(rows) {
  const out = { worker: rows.length, api: 0, vercel_functions: 0, static: 0, bytes: 0, api_bytes: 0 };
  for (const row of rows) {
    out[row.kind] += 1;
    out.bytes += row.bytes;
    if (row.kind === "api") out.api_bytes += row.bytes;
  }
  return out;
}

const browser = await chromium.launch();
const phases = [];
let records = [];
let started = 0;
try {
  const page = await (await browser.newContext()).newPage();
  page.on("response", async (response) => {
    const at = (Date.now() - started) / 1000;
    let bytes = 0;
    try {
      bytes = (await response.body()).length;
    } catch {
      // A redirect has no body.
    }
    records.push({ at, kind: kind(response.url()), bytes });
  });

  async function measure(name, path, guest) {
    records = [];
    started = Date.now();
    await page.goto(`${args.base}${path}`);
    // A reader scrolls now and then. Only pointerdown, keydown, wheel, and touchstart count as
    // activity (lib/session/store.ts), and an idle guest is signed out at 5 minutes.
    for (let waited = 0; waited < seconds; waited += 30) {
      await page.waitForTimeout(Math.min(30, seconds - waited) * 1000);
      await page.mouse.move(400, 400);
      await page.mouse.wheel(0, waited % 60 === 0 ? 200 : -200);
    }
    const ended = Date.now();
    // Measured only while seated: a page that lost its session polls far less.
    const signedIn = !guest || (await page.getByRole("button", { name: "Sign out" }).first().isVisible());
    const load = sum(records.filter((row) => row.at < loadSeconds));
    const steady = sum(records.filter((row) => row.at >= loadSeconds));
    const minutes = (seconds - loadSeconds) / 60;
    phases.push({
      name,
      guest,
      signed_in_at_end: signedIn,
      started_ms: started,
      ended_ms: ended,
      load,
      steady_per_minute: Object.fromEntries(
        Object.entries(steady).map(([k, v]) => [k, Math.round((v / minutes) * 10) / 10]),
      ),
    });
  }

  await measure("landing", "/", false);
  await page.goto(`${args.base}/queue`);
  await page.getByRole("button", { name: /Continue as guest/ }).first().click({ timeout: 60_000 });
  await page.getByRole("button", { name: "Sign out" }).first().waitFor({ timeout: 60_000 });
  const paths = args.paths.split(",");
  if (args.edit) paths.push(`/edits/${args.edit}`);
  for (const path of paths) await measure(path, path, true);
  await page.getByRole("button", { name: "Sign out" }).first().click({ timeout: 5000 }).catch(() => {});
} finally {
  await browser.close();
}
console.log(`RESULT ${JSON.stringify({ seconds, load_seconds: loadSeconds, phases })}`);
