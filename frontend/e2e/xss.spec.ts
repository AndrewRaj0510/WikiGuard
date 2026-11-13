/**
 * The XSS corpus, through the real UI (PLAN section 8). Vandals write titles, edit summaries, and
 * diff text, and the LLM's answer is untrusted too. Every payload must reach the page as visible
 * text and nothing else: no element made, no handler run, no dialog, and no CSP violation (which
 * the fixture checks for every test).
 */
import { expect, signInAsGuest, test } from "./fixtures";
import type { MockEdit } from "./mock-api";

const PAYLOADS = [
  `<script>window.__xss = "script"</script>`,
  `<img src=x onerror="window.__xss='img'">`,
  `<svg onload="window.__xss='svg'"></svg>`,
  `"><iframe src="javascript:window.__xss='iframe'"></iframe>`,
  `<a href="javascript:window.__xss='link'">click me</a>`,
  `[[javascript:alert(1)|wiki link]] **bold** _emphasis_ <b>tag</b>`,
  `{{Template:${"<"}style>body{display:none}</style>}}`,
  `‮evil right-to-left override`,
];

const EDITS: MockEdit[] = PAYLOADS.map((payload, index) => ({
  rev_id: 900 + index,
  title: payload,
  comment: payload,
  added_text: `${payload}\nsecond line ${payload}`,
  removed_text: payload,
  score: 0.9 - index * 0.05,
  is_anonymous: index % 2 === 0,
  minutes_ago: index + 1,
}));

test.use({ apiOptions: { edits: EDITS, explanation: PAYLOADS[1] } });

async function expectInert(page: import("@playwright/test").Page) {
  expect(await page.evaluate(() => (window as unknown as { __xss?: string }).__xss)).toBeUndefined();
  const injected = await page.evaluate(
    () => document.querySelectorAll("main iframe, main svg[onload], main img[onerror], main script").length,
  );
  expect(injected).toBe(0);
}

test("payloads in the queue and feed show as text", async ({ page, problems }) => {
  void problems;
  page.on("dialog", () => {
    throw new Error("A payload opened a dialog");
  });
  await signInAsGuest(page);
  for (const payload of PAYLOADS.slice(0, 5)) {
    await expect(page.getByRole("link", { name: payload, exact: true })).toBeVisible();
  }
  await expectInert(page);
  await page.getByRole("link", { name: "Edit feed" }).first().click();
  await expect(page.getByRole("list", { name: /Edit feed/ }).getByRole("listitem")).toHaveCount(PAYLOADS.length);
  await expectInert(page);
});

test("payloads on an edit's page, and in the LLM's answer, show as text", async ({ page, problems }) => {
  void problems;
  page.on("dialog", () => {
    throw new Error("A payload opened a dialog");
  });
  await signInAsGuest(page);
  for (const edit of EDITS) {
    await page.goto(`/edits/${edit.rev_id}`);
    await expect(page.getByRole("heading", { level: 1 })).toHaveText(edit.title);
    await expectInert(page);
    // Links out are built from the integer, so no payload can steer them.
    for (const href of await page.locator("a[target=_blank]").evaluateAll((links) => links.map((l) => l.getAttribute("href")))) {
      expect(href).toMatch(/^https:\/\/en\.wikipedia\.org\/w\/index\.php\?(diff=\d+|title=)/);
    }
  }
  await page.getByRole("button", { name: "Explain this edit" }).click();
  await expect(page.locator("blockquote")).toHaveText(PAYLOADS[1]);
  await expectInert(page);
});
