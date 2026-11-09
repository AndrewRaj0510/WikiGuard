/**
 * Screenshot every page of the app to docs/images/, as a guest and as the owner, in light, dark,
 * and mobile layouts (PLAN section 17). It also fails if any page logs a console error or a CSP
 * violation, so a capture run doubles as a check that nothing on screen is broken.
 *
 * Needs a running app with real data behind it, not mocks: screenshots show real results only.
 *   python scripts/fixture_stack.py                       # API on 127.0.0.1:8765
 *   cd frontend && API_ORIGIN=http://127.0.0.1:8765 npm run build && npm start
 *   node scripts/capture_screenshots.mjs --password-file data/fixture_stack/user_password.txt
 *
 * The owner password is read from a file and never printed. Check every image for IPs, usernames,
 * secrets, and offensive text before copying it to docs/images/readme/ (PLAN section 17).
 * The guest wait is per address, so a full run takes a few minutes.
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
    "password-file": { type: "string" },
    day: { type: "string", default: new Date().toISOString().slice(0, 10) },
  },
});
const day = args.day;
const out = resolve(args.out);
mkdirSync(out, { recursive: true });
const problems = [];
const saved = [];

const LAYOUTS = [
  { name: "light", colorScheme: "light", viewport: { width: 1360, height: 900 } },
  { name: "dark", colorScheme: "dark", viewport: { width: 1360, height: 900 } },
  { name: "mobile", colorScheme: "light", viewport: { width: 390, height: 844 } },
];

function watch(page, label) {
  page.on("console", (message) => {
    const text = message.text();
    // Expected: the first /auth/me before signing in answers 401, and a refused guest 429s.
    if (message.type() === "error" && !/status of (401|409|429)/.test(text)) {
      problems.push(`${label}: console error: ${text.slice(0, 200)}`);
    }
    if (/Content Security Policy/i.test(text)) problems.push(`${label}: CSP: ${text.slice(0, 200)}`);
  });
  page.on("pageerror", (error) => problems.push(`${label}: page error: ${error.message}`));
}

async function shot(page, name, layout) {
  await page.waitForTimeout(700);
  const file = join(out, `app_${name}_${layout.name}_${day}.png`);
  await page.screenshot({ path: file, fullPage: true });
  saved.push(file);
}

/**
 * Signs in as a guest, waiting out the gate when it says no: a previous run's seat lasts 45 s, and
 * a guest sign-out makes this address wait 60 s. That's the gate working, so wait, don't fail.
 */
async function signInAsGuest(page) {
  const queue = page.getByRole("list", { name: /Review queue/ });
  for (let tries = 0; tries < 20; tries += 1) {
    await page.getByRole("button", { name: "Continue as guest" }).first().click();
    const seated = await queue.waitFor({ timeout: 8000 }).then(() => true, () => false);
    if (seated) return;
    await page.waitForTimeout(10_000);
    await page.goto(`${args.base}/queue`);
  }
  throw new Error("The gate never let a guest in");
}

async function guestRun(browser, layout) {
  const context = await browser.newContext({ colorScheme: layout.colorScheme, viewport: layout.viewport });
  const page = await context.newPage();
  watch(page, `guest/${layout.name}`);

  await page.goto(`${args.base}/`);
  await shot(page, "landing", layout);
  if (layout.name === "light") {
    await page.goto(`${args.base}/privacy`);
    await shot(page, "privacy", layout);
  }

  await page.goto(`${args.base}/queue`);
  await page.getByRole("button", { name: "Continue as guest" }).waitFor();
  if (layout.name === "light") {
    await page.getByRole("button", { name: "Owner sign-in" }).click();
    await shot(page, "sign_in", layout);
  }
  await signInAsGuest(page);
  await shot(page, "queue", layout);

  await page.locator("label", { hasText: "Logged out" }).click();
  await page.waitForTimeout(1200);
  await shot(page, "queue_logged_out", layout);

  await page.getByRole("link", { name: "Live feed" }).first().click();
  await page.getByRole("list", { name: /Live feed/ }).waitFor();
  await shot(page, "feed", layout);

  await page.getByRole("list", { name: /Live feed/ }).getByRole("link").first().click();
  await page.getByRole("heading", { level: 1 }).waitFor();
  await shot(page, "edit_guest", layout);

  await page.getByRole("link", { name: "Models" }).first().click();
  await page.getByRole("table").first().waitFor();
  await page.waitForTimeout(1500); // the chart loads on demand
  await shot(page, "models", layout);

  await page.getByRole("link", { name: "RoBERTa" }).first().click();
  await page.getByRole("heading", { name: /RoBERTa/ }).waitFor();
  await page.waitForTimeout(1500);
  await shot(page, "model_card", layout);

  await page.getByRole("link", { name: "Health" }).first().click();
  await page.getByRole("heading", { name: "Health" }).waitFor();
  await page.waitForTimeout(1000);
  await shot(page, "health_guest", layout);

  // Leave the seat free for the owner's run.
  await page.getByRole("button", { name: "Sign out" }).click();
  await context.close();
}

async function ownerRun(browser, layout, password) {
  const context = await browser.newContext({ colorScheme: layout.colorScheme, viewport: layout.viewport });
  const page = await context.newPage();
  watch(page, `owner/${layout.name}`);
  await page.goto(`${args.base}/queue`);
  await page.getByRole("button", { name: "Owner sign-in" }).click();
  await page.getByLabel("Owner password").fill(password);
  await page.getByRole("button", { name: "Sign in" }).click();
  const takeOver = page.getByRole("button", { name: "Take the seat" });
  if (await takeOver.isVisible({ timeout: 3000 }).catch(() => false)) await takeOver.click();
  await page.getByRole("list", { name: /Review queue/ }).waitFor({ timeout: 30_000 });

  // The keyboard flow: j to the first row, Enter to open it.
  await page.keyboard.press("j");
  await page.keyboard.press("Enter");
  await page.getByRole("heading", { name: "Your decision" }).waitFor();
  await shot(page, "edit_owner", layout);

  await page.keyboard.press("?");
  await page.getByRole("dialog").waitFor();
  await shot(page, "shortcuts", layout);
  await page.keyboard.press("Escape");

  await page.getByRole("link", { name: "Health" }).first().click();
  await page.getByRole("heading", { name: "Owner controls" }).waitFor();
  await shot(page, "health_owner", layout);

  await page.getByRole("button", { name: "Sign out" }).click();
  await context.close();
}

const browser = await chromium.launch();
try {
  for (const layout of LAYOUTS) await guestRun(browser, layout);
  if (args["password-file"]) {
    const password = readFileSync(args["password-file"], "utf-8").trim();
    await ownerRun(browser, LAYOUTS[0], password);
  } else {
    console.log("No --password-file, so the owner's pages were skipped.");
  }
} finally {
  await browser.close();
}

console.log(`Saved ${saved.length} screenshots to ${out}`);
if (problems.length) {
  console.error(`${problems.length} problem(s):\n${problems.join("\n")}`);
  process.exit(1);
}
