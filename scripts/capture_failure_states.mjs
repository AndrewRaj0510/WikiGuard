/**
 * Watch one page of the app through a failure, for scripts/inject_failures.py (Phase 8).
 *
 * It opens PATH (signed in as the owner when --password-file is given), then works through the
 * --step list in order. After each step it saves a screenshot and prints "STEP n <seconds>", which
 * is the injection script's cue to break or mend the next thing. Steps:
 *
 *   text:REGEX    wait until the page shows text matching REGEX
 *   gone:REGEX    wait until it no longer does
 *   click:NAME    click the button with that accessible name
 *
 * Errors in the console are expected here (the API is down on purpose), so they're counted and
 * reported, not treated as failures. A CSP violation still fails the run: no failure excuses one.
 * The last line is "RESULT {json}". The page polls on its own; nothing here reloads it, because
 * recovering without a reload is part of what's being tested.
 *
 *   node scripts/capture_failure_states.mjs --path /queue --name api_down \
 *     --step "text:isn.t answering" --step "gone:isn.t answering"
 */
import { createRequire } from "node:module";
import { mkdirSync, readFileSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { parseArgs } from "node:util";

const root = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const require = createRequire(join(root, "frontend", "package.json"));
const { chromium } = require("@playwright/test");

const { values: args } = parseArgs({
  options: {
    base: { type: "string", default: "http://localhost:3000" },
    out: { type: "string", default: join(root, "docs", "images") },
    path: { type: "string", default: "/queue" },
    name: { type: "string" },
    "password-file": { type: "string" },
    step: { type: "string", multiple: true, default: [] },
    "timeout-seconds": { type: "string", default: "900" },
    day: { type: "string", default: new Date().toISOString().slice(0, 10) },
  },
});
if (!args.name) throw new Error("--name is required");
mkdirSync(resolve(args.out), { recursive: true });
const timeoutMs = Number(args["timeout-seconds"]) * 1000;
const started = Date.now();
const elapsed = () => Math.round((Date.now() - started) / 100) / 10;
const result = { name: args.name, steps: [], console_errors: 0, csp_violations: [], screenshots: [] };

function pattern(source) {
  return new RegExp(source, "i");
}

async function waitFor(page, check) {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    const text = await page.locator("body").innerText().catch(() => "");
    if (check(text)) return true;
    await page.waitForTimeout(1000);
  }
  return false;
}

async function signIn(page, password) {
  await page.getByRole("button", { name: "Owner sign-in" }).click({ timeout: 120_000 });
  await page.getByLabel("Owner password").fill(password);
  await page.getByRole("button", { name: "Sign in" }).click();
  const takeOver = page.getByRole("button", { name: "Take the seat" });
  if (await takeOver.isVisible({ timeout: 3000 }).catch(() => false)) await takeOver.click();
  await page.getByRole("button", { name: "Sign out" }).first().waitFor({ timeout: 60_000 });
}

const browser = await chromium.launch();
let page;
try {
  const context = await browser.newContext({ viewport: { width: 1360, height: 900 } });
  page = await context.newPage();
  page.on("console", (message) => {
    const text = message.text();
    if (/Content Security Policy/i.test(text)) result.csp_violations.push(text.slice(0, 200));
    else if (message.type() === "error") result.console_errors += 1;
  });
  await page.goto(`${args.base}${args.path}`);
  if (args["password-file"]) {
    await signIn(page, readFileSync(args["password-file"], "utf-8").trim());
    await page.goto(`${args.base}${args.path}`);
  }
  console.log(`READY ${elapsed()}`);

  for (const [index, step] of args.step.entries()) {
    const [kind, ...rest] = step.split(":");
    const value = rest.join(":");
    let ok = true;
    if (kind === "click") {
      await page.getByRole("button", { name: value }).first().click({ timeout: timeoutMs });
    } else if (kind === "text") {
      ok = await waitFor(page, (text) => pattern(value).test(text));
    } else if (kind === "gone") {
      ok = await waitFor(page, (text) => !pattern(value).test(text));
    } else {
      throw new Error(`unknown step ${step}`);
    }
    await page.waitForTimeout(500);
    const file = join(resolve(args.out), `failure_${args.name}_${index + 1}_${args.day}.png`);
    await page.screenshot({ path: file, fullPage: true });
    result.screenshots.push(file);
    result.steps.push({ step, ok, seconds: elapsed() });
    console.log(`STEP ${index + 1} ${ok ? "ok" : "timeout"} ${elapsed()}`);
    if (!ok) break;
  }
  // Free the seat, so the next run doesn't wait 45 s for it. Fine to fail when the API is down.
  await page.getByRole("button", { name: "Sign out" }).first().click({ timeout: 3000 }).catch(() => {});
} finally {
  await browser.close();
}
console.log(`RESULT ${JSON.stringify(result)}`);
process.exit(result.csp_violations.length ? 1 : 0);
