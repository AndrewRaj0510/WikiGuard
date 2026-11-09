import { test as base, expect, type Page } from "@playwright/test";
import { MockApi, OWNER_PASSWORD, type MockOptions } from "./mock-api";

/**
 * Every test gets a fresh fake API, and fails if the page logs an error or breaks its CSP. The
 * expected noise is filtered out: the 401 from /auth/me before signing in.
 */
export const test = base.extend<{ api: MockApi; apiOptions: MockOptions; problems: string[] }>({
  apiOptions: [{}, { option: true }],
  // auto: every test gets the mock, even one that never reads it, so no request ever escapes to
  // a real server.
  api: [
    async ({ page, apiOptions }, use) => {
      const api = new MockApi(apiOptions);
      await api.install(page);
      await use(api);
    },
    { auto: true },
  ],
  problems: async ({ page }, use) => {
    const problems: string[] = [];
    page.on("console", (message) => {
      const text = message.text();
      if (/Content Security Policy/i.test(text)) problems.push(`CSP: ${text}`);
      // Expected: refused answers the tests ask for, and requests the down-API tests abort.
      else if (message.type() === "error" && !/status of (401|403|404|409|429)|net::ERR_/.test(text)) {
        problems.push(text);
      }
    });
    page.on("pageerror", (error) => problems.push(`page error: ${error.message}`));
    await use(problems);
    expect(problems, "console errors or CSP violations").toEqual([]);
  },
});

export { expect };

export async function signInAsGuest(page: Page): Promise<void> {
  await page.goto("/queue");
  await page.getByRole("button", { name: "Continue as guest" }).click();
  await expect(page.getByRole("list", { name: /Review queue/ })).toBeVisible();
}

export async function signInAsOwner(page: Page): Promise<void> {
  await page.goto("/queue");
  await page.getByRole("button", { name: "Owner sign-in" }).click();
  await page.getByLabel("Owner password").fill(OWNER_PASSWORD);
  await page.getByRole("button", { name: "Sign in" }).click();
  await expect(page.getByRole("list", { name: /Review queue/ })).toBeVisible();
}
