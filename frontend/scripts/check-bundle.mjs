/**
 * Two checks on a production build, for CI (PLAN sections 8 and 13):
 *
 * 1. No secrets in anything a browser can download (.next/static). It looks for the shapes of real
 *    keys, and for the values of server-only settings that happen to be set while it runs, so a
 *    secret that leaked through a NEXT_PUBLIC_ variable or an import would be caught.
 * 2. A JavaScript budget per page: it starts the built app, loads each page's HTML, and adds up
 *    the gzipped size of every script that page loads first. Charts load later, on demand.
 *
 *   npm run build && npm run check:bundle
 */
import { spawn } from "node:child_process";
import { readdirSync, readFileSync, statSync } from "node:fs";
import { join, resolve } from "node:path";
import { gzipSync } from "node:zlib";

const root = resolve(import.meta.dirname, "..");
const staticDir = join(root, ".next", "static");

// --- 1. Secrets ---

export const SECRET_PATTERNS = [
  // Not after a letter: Tailwind's own class names include "mask-linear-…" (F-117).
  ["OpenAI key", /(?<![\w-])sk-(?:proj-|svcacct-)?[A-Za-z0-9_-]{32,}/],
  ["Google API key", /AIza[0-9A-Za-z_-]{35}/],
  ["GitHub token", /gh[pousr]_[A-Za-z0-9]{36,}/],
  ["private key", /-----BEGIN [A-Z ]*PRIVATE KEY-----/],
  ["Postgres URL with a password", /postgres(?:ql)?:\/\/[^\s:"'/]+:[^\s@"'/]+@/],
  ["Argon2 hash", /\$argon2(?:id|i|d)\$/],
  ["JWT", /eyJ[A-Za-z0-9_-]{10,}\.eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}/],
];

// Names from .env.example that must never reach a browser. Only their values are checked.
export const SERVER_ONLY = [
  "DATABASE_URL",
  "DATABASE_MIGRATIONS_URL",
  "SESSION_SECRET_KEY",
  "USER_PASSWORD_HASH",
  "METRICS_BEARER_TOKEN",
  "USER_HASH_KEY",
  "OPENAI_API_KEY",
  "GEMINI_API_KEY",
  "EDGE_SHARED_SECRET",
  "MLFLOW_TRACKING_PASSWORD",
  "POSTGRES_PASSWORD",
  // Not a secret, but the design says the browser never learns the API's real address.
  "API_ORIGIN",
];

export function findSecrets(text, values = []) {
  const found = [];
  for (const [name, pattern] of SECRET_PATTERNS) if (pattern.test(text)) found.push(name);
  for (const [name, value] of values) if (value && text.includes(value)) found.push(`the value of ${name}`);
  return found;
}

function files(dir) {
  return readdirSync(dir).flatMap((name) => {
    const path = join(dir, name);
    return statSync(path).isDirectory() ? files(path) : [path];
  });
}

function checkSecrets() {
  const values = SERVER_ONLY.map((name) => [name, process.env[name]])
    .filter(([, value]) => value && value.length >= 8);
  const leaks = [];
  for (const file of files(staticDir)) {
    if (!/\.(js|css|json|html|txt|map)$/.test(file)) continue;
    for (const hit of findSecrets(readFileSync(file, "utf-8"), values)) {
      leaks.push(`${file.slice(root.length + 1)}: ${hit}`);
    }
  }
  const publicVars = Object.keys(process.env).filter((name) => name.startsWith("NEXT_PUBLIC_"));
  console.log(`Secrets: scanned ${files(staticDir).length} files, checked ${values.length} server-only values.`);
  if (publicVars.length) console.log(`NEXT_PUBLIC_ variables set (they reach the browser): ${publicVars.join(", ")}`);
  return leaks;
}

// --- 2. Budgets (gzipped first-load JavaScript, KB) ---

// About 10% above what was measured on 2026-11-09 (docs/metrics.md). Raise one only on purpose.
export const BUDGETS_KB = {
  "/": 212,
  "/privacy": 208,
  "/queue": 245,
  "/feed": 245,
  "/edits/1": 245,
  "/models": 245,
  "/models/roberta": 245,
  "/health": 245,
};

const PORT = 3299;

async function waitForServer(url, ms = 30_000) {
  const end = Date.now() + ms;
  while (Date.now() < end) {
    try {
      if ((await fetch(url)).ok) return;
    } catch {
      // not yet
    }
    await new Promise((r) => setTimeout(r, 500));
  }
  throw new Error("The app didn't start");
}

async function measure() {
  const server = spawn(process.execPath, [join(root, "node_modules", "next", "dist", "bin", "next"), "start", "-p", String(PORT)], {
    cwd: root,
    stdio: "ignore",
  });
  const base = `http://localhost:${PORT}`;
  const results = [];
  try {
    await waitForServer(`${base}/privacy`);
    for (const [path, budget] of Object.entries(BUDGETS_KB)) {
      const html = await (await fetch(base + path)).text();
      const scripts = [...new Set([...html.matchAll(/<script[^>]+src="([^"]+\.js)"/g)].map((m) => m[1]))];
      let bytes = 0;
      for (const src of scripts) {
        bytes += gzipSync(Buffer.from(await (await fetch(base + src)).arrayBuffer()), { level: 9 }).length;
      }
      results.push({ path, scripts: scripts.length, kb: bytes / 1024, budget });
    }
  } finally {
    server.kill();
  }
  return results;
}

if (process.argv[1] && resolve(process.argv[1]) !== resolve(import.meta.filename)) {
  // Imported by a test: export only.
} else {
  await main();
}

async function main() {
const leaks = checkSecrets();
const results = await measure();
console.log("\nFirst-load JavaScript, gzipped:");
for (const r of results) {
  const mark = r.kb > r.budget ? "OVER" : "ok";
  console.log(`  ${mark.padEnd(4)} ${r.path.padEnd(18)} ${r.kb.toFixed(1).padStart(6)} KB of ${r.budget} KB (${r.scripts} files)`);
}
const over = results.filter((r) => r.kb > r.budget);
if (leaks.length) console.error(`\nPossible secrets in the client bundle:\n  ${leaks.join("\n  ")}`);
if (over.length) console.error(`\n${over.length} page(s) over budget.`);
process.exit(leaks.length || over.length ? 1 : 0);
}
