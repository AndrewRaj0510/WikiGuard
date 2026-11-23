/**
 * What a browser can see of WikiGuard's secrets: nothing (PLAN.md, Phase 8; scripts/check_logs.py).
 *
 * Signs in as the owner through the real Worker (scripts/edge_local.mjs) in Chromium, opens every
 * console page, and then checks:
 *   - the session cookie is HttpOnly and SameSite, so page scripts can't read it: it's missing
 *     from document.cookie;
 *   - nothing is kept in localStorage or sessionStorage: not the session, not the CSRF token;
 *   - no response the browser received (HTML, scripts, styles, JSON) contains any secret value,
 *     the session cookie's value, or the name of a server-only setting;
 *   - every request went to the site's own origin, never to the API's origin or anywhere else.
 *
 *   node scripts/check_browser_secrets.mjs --base http://localhost:8787 --edit 123
 * The owner's password comes from the file PASSWORD_FILE names, and the secret values to look for
 * from the JSON list in SECRETS_FILE: files, so no secret is on a command line. The last line is
 * "RESULT {json}", which names what was found, never the values.
 */
import { createRequire } from "node:module";
import { readFileSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { parseArgs } from "node:util";

const root = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const require = createRequire(join(root, "frontend", "package.json"));
const { chromium } = require("@playwright/test");

const { values: args } = parseArgs({
  options: {
    base: { type: "string", default: "http://localhost:8787" },
    edit: { type: "string" },
  },
});
const secrets = JSON.parse(readFileSync(process.env.SECRETS_FILE, "utf-8"));
const SETTING_NAMES = [
  "SESSION_SECRET_KEY",
  "USER_PASSWORD_HASH",
  "USER_HASH_KEY",
  "METRICS_BEARER_TOKEN",
  "EDGE_SHARED_SECRET",
  "OPENAI_API_KEY",
  "GEMINI_API_KEY",
  "DATABASE_URL",
];
const SESSION_COOKIE = "wikiguard_session";
const origin = new URL(args.base).origin;
const bodies = [];
const foreignRequests = new Set();
let csrfToken = null;

const browser = await chromium.launch();
const result = { pages: [], responses_checked: 0 };
try {
  const context = await browser.newContext();
  const page = await context.newPage();
  page.on("request", (request) => {
    const url = new URL(request.url());
    if (url.origin !== origin && !url.protocol.startsWith("data")) foreignRequests.add(url.origin);
  });
  page.on("response", async (response) => {
    const type = response.headers()["content-type"] ?? "";
    if (!/text|javascript|json|css/.test(type)) return;
    try {
      const text = await response.text();
      bodies.push({ url: response.url(), text });
      if (response.url().endsWith("/api/v1/auth/login")) csrfToken = JSON.parse(text).csrf_token;
    } catch {
      // A redirect or an aborted request has no body to read.
    }
  });

  await page.goto(`${args.base}/queue`);
  await page.getByRole("button", { name: "Owner sign-in" }).click({ timeout: 60_000 });
  await page.getByLabel("Owner password").fill(readFileSync(process.env.PASSWORD_FILE, "utf-8").trim());
  await page.getByRole("button", { name: "Sign in" }).click();
  const takeOver = page.getByRole("button", { name: "Take the seat" });
  if (await takeOver.isVisible({ timeout: 3000 }).catch(() => false)) await takeOver.click();
  await page.getByRole("button", { name: "Sign out" }).first().waitFor({ timeout: 60_000 });

  for (const path of ["/", "/queue", "/feed", `/edits/${args.edit}`, "/models", "/health", "/privacy"]) {
    await page.goto(`${args.base}${path}`);
    await page.waitForTimeout(3000);
    result.pages.push(path);
  }

  const cookies = await context.cookies();
  const session = cookies.find((cookie) => cookie.name === SESSION_COOKIE);
  const seen = await page.evaluate(() => ({
    documentCookie: document.cookie,
    local: Object.entries(localStorage),
    session: Object.entries(sessionStorage),
  }));
  const storageText = JSON.stringify([seen.local, seen.session]);

  result.session_cookie = session
    ? { http_only: session.httpOnly, same_site: session.sameSite, secure: session.secure }
    : null;
  result.session_cookie_in_document_cookie = seen.documentCookie.includes(SESSION_COOKIE);
  result.storage_keys = [...seen.local, ...seen.session].map(([key]) => key);
  result.session_value_in_storage = Boolean(session) && storageText.includes(session.value);
  result.csrf_token_seen = Boolean(csrfToken);
  result.csrf_token_in_storage_or_cookie =
    Boolean(csrfToken) && (storageText.includes(csrfToken) || seen.documentCookie.includes(csrfToken));

  // What came back to the browser: no secret value, no session value, no server-only name.
  const hits = new Set();
  for (const { url, text } of bodies) {
    secrets.forEach((value, index) => {
      if (value && text.includes(value)) hits.add(`secret #${index} in ${new URL(url).pathname}`);
    });
    if (session && text.includes(session.value)) hits.add(`session value in ${new URL(url).pathname}`);
    for (const name of SETTING_NAMES) {
      if (text.includes(name)) hits.add(`${name} named in ${new URL(url).pathname}`);
    }
  }
  result.responses_checked = bodies.length;
  result.found_in_responses = [...hits];
  result.foreign_origins = [...foreignRequests];
  await page.getByRole("button", { name: "Sign out" }).first().click({ timeout: 5000 }).catch(() => {});
} finally {
  await browser.close();
}
console.log(`RESULT ${JSON.stringify(result)}`);
