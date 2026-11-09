import { defineConfig, devices } from "@playwright/test";

/**
 * End-to-end tests against the production build, with the API mocked in the browser
 * (e2e/mock-api.ts). Build first: `npm run build`, then `npm run test:e2e`.
 */
const PORT = 3210;

export default defineConfig({
  testDir: "e2e",
  fullyParallel: true,
  // Every page renders per request (the CSP nonce), on one `next start` process: more workers
  // than this overload it on a laptop or a CI runner, and tests time out on load, not on logic.
  workers: 2,
  timeout: 60_000,
  forbidOnly: Boolean(process.env.CI),
  retries: process.env.CI ? 1 : 0,
  reporter: process.env.CI ? [["github"], ["list"]] : "list",
  use: {
    baseURL: `http://localhost:${PORT}`,
    trace: "retain-on-failure",
  },
  projects: [{ name: "chromium", use: { ...devices["Desktop Chrome"] } }],
  webServer: {
    command: `npm run start -- -p ${PORT}`,
    url: `http://localhost:${PORT}/privacy`,
    reuseExistingServer: !process.env.CI,
    timeout: 60_000,
  },
});
