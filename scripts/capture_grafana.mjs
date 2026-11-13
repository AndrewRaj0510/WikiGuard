/**
 * Screenshot every WikiGuard Grafana dashboard to docs/images/ (PLAN section 17). Grafana stays
 * private (F-037), so it's reached through a port-forward or the tunnel, never the internet:
 *
 *   kubectl -n monitoring port-forward svc/kps-grafana 13000:80
 *   node scripts/capture_grafana.mjs --password-file data/kind/grafana_password.txt
 *
 * The admin password is read from a file and never printed. It also fails if a panel shows an
 * error, so a capture doubles as a check that every query works against the real data sources.
 * Check every image for usernames, addresses, secrets, and offensive text before copying one to
 * docs/images/readme/.
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
    base: { type: "string", default: "http://127.0.0.1:13000" },
    out: { type: "string", default: join(root, "docs", "images") },
    "password-file": { type: "string" },
    range: { type: "string", default: "now-3h" },
    day: { type: "string", default: new Date().toISOString().slice(0, 10) },
  },
});
const DASHBOARDS = ["pipeline", "models", "api", "security", "free-tier"];
const out = resolve(args.out);
mkdirSync(out, { recursive: true });
const password = readFileSync(args["password-file"], "utf-8").trim();

const browser = await chromium.launch();
// Tall enough for a whole dashboard: Grafana only draws the panels inside the viewport.
const context = await browser.newContext({ viewport: { width: 1600, height: 2000 }, colorScheme: "dark" });
const page = await context.newPage();
await page.goto(`${args.base}/login`);
await page.getByTestId("data-testid Username input field").fill("admin");
await page.getByTestId("data-testid Password input field").fill(password);
await page.getByTestId("data-testid Login button").click();
await page.waitForURL((url) => !url.pathname.startsWith("/login"));

const problems = [];
for (const name of DASHBOARDS) {
  await page.goto(`${args.base}/d/wikiguard-${name}?orgId=1&from=${args.range}&to=now&kiosk`);
  await page.waitForLoadState("networkidle");
  // Wait until the panels have drawn: charts render to canvas or SVG, stats and tables to text.
  await page
    .waitForFunction(() => document.querySelectorAll("canvas, .u-over, [data-testid='data-testid table body']").length > 0, null, { timeout: 30000 })
    .catch(() => problems.push(`${name}: no panel drew anything in 30 s`));
  await page.waitForTimeout(6000);
  const errors = await page.locator('[data-testid="data-testid Panel status error"]').count();
  if (errors) problems.push(`${name}: ${errors} panel(s) show an error`);
  const file = join(out, `grafana_${name}_${args.day}.png`);
  await page.screenshot({ path: file, fullPage: true });
  console.log(`saved ${file}`);
}
await browser.close();
if (problems.length) {
  console.error(problems.join("\n"));
  process.exit(1);
}
