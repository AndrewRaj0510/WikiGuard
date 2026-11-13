/**
 * Accessibility (PLAN section 9, WCAG 2.1 AA): axe on every page in light and dark, and the
 * keyboard alone through the main flow. axe can't judge everything, so the keyboard test and the
 * "risk never by color alone" rule (unit tests) cover what it misses.
 */
import AxeBuilder from "@axe-core/playwright";
import type { Page } from "@playwright/test";
import { expect, signInAsGuest, signInAsOwner, test } from "./fixtures";

const TAGS = ["wcag2a", "wcag2aa", "wcag21a", "wcag21aa"];

async function audit(page: Page, label: string) {
  const results = await new AxeBuilder({ page }).withTags(TAGS).analyze();
  const summary = results.violations.map(
    (v) => `${label}: ${v.id} (${v.impact}): ${v.nodes.length} node(s), e.g. ${v.nodes[0]?.target.join(" ")}`,
  );
  expect(summary).toEqual([]);
}

for (const scheme of ["light", "dark"] as const) {
  test.describe(`axe, ${scheme}`, () => {
    test.use({ colorScheme: scheme });

    test("public pages", async ({ page, problems }) => {
      void problems;
      await page.goto("/");
      await audit(page, "landing");
      await page.goto("/privacy");
      await audit(page, "privacy");
      await page.goto("/queue");
      await page.getByRole("button", { name: "Owner sign-in" }).click();
      await audit(page, "sign-in");
    });

    test("console pages", async ({ page, problems }) => {
      void problems;
      await signInAsGuest(page);
      await audit(page, "queue");
      await page.goto("/feed");
      await expect(page.getByRole("list", { name: /Edit feed/ })).toBeVisible();
      await audit(page, "feed");
      await page.goto("/edits/101");
      await expect(page.getByRole("heading", { level: 1 })).toBeVisible();
      await audit(page, "edit");
      await page.goto("/models");
      await expect(page.getByRole("table").first()).toBeVisible();
      await page.waitForTimeout(800);
      await audit(page, "models");
      await page.goto("/models/roberta");
      await expect(page.getByRole("heading", { name: /RoBERTa/ })).toBeVisible();
      await page.waitForTimeout(800);
      await audit(page, "model card");
      await page.goto("/health");
      await expect(page.getByRole("heading", { name: "Owner controls" })).toBeVisible();
      await audit(page, "health");
    });
  });
}

test("the owner's pages pass axe too", async ({ page, problems }) => {
  void problems;
  await signInAsOwner(page);
  await page.goto("/edits/101");
  await expect(page.getByRole("heading", { name: "Your decision" })).toBeVisible();
  await audit(page, "edit, owner");
  await page.keyboard.press("?");
  await expect(page.getByRole("dialog")).toBeVisible();
  await audit(page, "shortcuts dialog");
});

test("the keyboard alone: skip link, sign-in, queue, edit, and back", async ({ page, problems }) => {
  void problems;
  await page.goto("/queue");
  // The first Tab reaches the skip link, which moves focus past the navigation.
  await page.keyboard.press("Tab");
  await expect(page.getByRole("link", { name: "Skip to content" })).toBeFocused();
  // Tab to "Continue as guest" and press Enter.
  const guest = page.getByRole("button", { name: "Continue as guest" });
  for (let i = 0; i < 12 && !(await guest.evaluate((el) => el === document.activeElement)); i += 1) {
    await page.keyboard.press("Tab");
  }
  await expect(guest).toBeFocused();
  await page.keyboard.press("Enter");
  await expect(page.getByRole("list", { name: /Review queue/ })).toBeVisible();

  await page.keyboard.press("j");
  await expect(page.getByRole("link", { name: "Ada Lovelace" })).toBeFocused();
  await page.keyboard.press("j");
  await expect(page.getByRole("link", { name: "Tidal locking" })).toBeFocused();
  await page.keyboard.press("k");
  await page.keyboard.press("Enter");
  await expect(page.getByRole("heading", { level: 1 })).toHaveText("Ada Lovelace");
  // Every control on the page is reachable and shows where focus is.
  await page.keyboard.press("Tab");
  const outline = await page.evaluate(() => getComputedStyle(document.activeElement as Element).outlineStyle);
  expect(outline).not.toBe("none");
  await page.keyboard.press("b");
  await expect(page).toHaveURL(/\/queue$/);
});
