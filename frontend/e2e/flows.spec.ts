import { expect, signInAsGuest, signInAsOwner, test } from "./fixtures";

test.describe("feed and queue", () => {
  test("the queue ranks by score and filters logged-out edits on the server", async ({ page, api, problems }) => {
    void problems;
    await signInAsGuest(page);
    const rows = page.getByRole("list", { name: /Review queue/ }).getByRole("listitem");
    await expect(rows.first()).toContainText("Ada Lovelace");
    await expect(rows.first()).toContainText("Logged out");

    await page.locator("label", { hasText: "Logged out" }).click();
    await expect(page).toHaveURL(/editor=logged_out/);
    await expect(rows).toHaveCount(2);
    expect(api.calls.some((c) => c.path === "/v1/queue" && c.method === "GET")).toBe(true);
  });

  test("the live feed shows a new edit when it arrives", async ({ page, api, problems }) => {
    void problems;
    await signInAsGuest(page);
    await page.getByRole("link", { name: "Live feed" }).first().click();
    const feed = page.getByRole("list", { name: /Live feed/ });
    await expect(feed.getByRole("listitem")).toHaveCount(4);
    api.arrive({
      rev_id: 200,
      title: "Fresh arrival",
      comment: "new",
      added_text: "x",
      removed_text: "",
      score: 0.2,
      is_anonymous: false,
      minutes_ago: 0,
    });
    // The feed polls every 3 s.
    await expect(feed.getByRole("listitem").first()).toContainText("Fresh arrival", { timeout: 8000 });
  });
});

test.describe("review", () => {
  test("the owner reviews with the keyboard and moves to the next edit", async ({ page, api, problems }) => {
    void problems;
    await signInAsOwner(page);
    await page.keyboard.press("j");
    await page.keyboard.press("Enter");
    await expect(page.getByRole("heading", { level: 1 })).toHaveText("Ada Lovelace");

    await page.keyboard.press("v");
    await expect(page.getByRole("button", { name: /^Vandalism/ })).toHaveAttribute("aria-pressed", "true");
    const put = api.calls.find((c) => c.method === "PUT");
    expect(put?.path).toBe("/v1/edits/101/review");
    expect(put?.body).toEqual({ decision: "vandalism" });
    expect(put?.headers["x-csrf-token"]).toBeTruthy();

    await page.keyboard.press("n");
    await expect(page.getByRole("heading", { level: 1 })).toHaveText("Tidal locking");
    await page.keyboard.press("b");
    await expect(page).toHaveURL(/\/queue$/);
  });

  test("a guest sees the review controls switched off, and the server refuses anyway", async ({ page, api, problems }) => {
    void problems;
    await signInAsGuest(page);
    await page.getByRole("link", { name: "Ada Lovelace" }).click();
    await expect(page.getByText("Reviewing needs the owner’s account")).toBeVisible();
    await expect(page.getByRole("button", { name: /^Vandalism/ })).toBeDisabled();
    await page.keyboard.press("v");
    expect(api.calls.some((c) => c.method === "PUT")).toBe(false);
  });
});

test.describe("explain", () => {
  test("shows the answer as plain text and counts a guest's allowance", async ({ page, problems }) => {
    void problems;
    await signInAsGuest(page);
    await page.getByRole("link", { name: "Ada Lovelace" }).click();
    await expect(page.getByText("5 fresh left")).toBeVisible();
    await page.getByRole("button", { name: "Explain this edit" }).click();
    await expect(page.getByText("Changed the birth year from 1816 to 1815.")).toBeVisible();
    await expect(page.getByText("AI-written, may be wrong")).toBeVisible();
    await expect(page.getByText("4 fresh left")).toBeVisible();
  });
});

test.describe("when the API is down", () => {
  test("says so, keeps trying, and connects by itself once it's back", async ({ page, api, problems }) => {
    void problems;
    api.down = true;
    await page.goto("/queue");
    await expect(page.getByRole("heading", { name: "WikiGuard isn’t answering" })).toBeVisible();
    api.down = false;
    // /ready is polled every 5 s; once it answers, the session starts on its own.
    await expect(page.getByRole("button", { name: "Continue as guest" })).toBeVisible({ timeout: 12_000 });
  });

  test("keeps showing the last data and says it's reconnecting", async ({ page, api, problems }) => {
    void problems;
    await signInAsGuest(page);
    await page.getByRole("link", { name: "Live feed" }).first().click();
    await expect(page.getByRole("list", { name: /Live feed/ }).getByRole("listitem")).toHaveCount(4);
    api.down = true;
    await expect(page.getByText("Reconnecting")).toBeVisible({ timeout: 8000 });
    await expect(page.getByRole("list", { name: /Live feed/ }).getByRole("listitem")).toHaveCount(4);
  });
});

test.describe("models and health", () => {
  test("the leaderboard, a model card, and the health page load", async ({ page, problems }) => {
    void problems;
    await signInAsGuest(page);
    await page.getByRole("link", { name: "Models" }).first().click();
    await expect(page.getByRole("table").first()).toContainText("RoBERTa");
    await expect(page.getByLabel(/PR-AUC 0\.360, 95% interval/)).toBeVisible();
    await page.getByRole("link", { name: "RoBERTa" }).click();
    await expect(page.getByRole("heading", { name: /RoBERTa/ })).toBeVisible();
    await page.getByRole("link", { name: "Health" }).first().click();
    await expect(page.getByText("Not promoted. The interval includes zero.")).toBeVisible();
    await expect(page.getByRole("button", { name: "Request a retrain" })).toBeDisabled();
  });

  test("the owner can send a retrain request", async ({ page, api, problems }) => {
    void problems;
    await signInAsOwner(page);
    await page.getByRole("link", { name: "Health" }).first().click();
    await page.getByLabel("Reason").first().fill("weekly check");
    await page.getByRole("button", { name: "Request a retrain" }).click();
    await expect(page.getByText("Request 1 is pending.")).toBeVisible();
    expect(api.calls.find((c) => c.path === "/v1/admin/retrain")?.body).toEqual({ reason: "weekly check" });
  });
});

test.describe("pages and headers", () => {
  test("console pages carry a fresh CSP nonce and noindex; the landing page is indexable", async ({ request }) => {
    const first = await request.get("/queue");
    const second = await request.get("/queue");
    const nonce = (csp: string | undefined) => /'nonce-([^']+)'/.exec(csp ?? "")?.[1];
    expect(nonce(first.headers()["content-security-policy"])).toBeTruthy();
    expect(nonce(first.headers()["content-security-policy"])).not.toBe(nonce(second.headers()["content-security-policy"]));
    expect(first.headers()["x-robots-tag"]).toBe("noindex, nofollow");
    expect(first.headers()["x-frame-options"]).toBe("DENY");
    const landing = await request.get("/");
    expect(landing.headers()["x-robots-tag"]).toBeUndefined();
  });

  test("an unknown page gets the custom 404", async ({ page, problems }) => {
    void problems;
    const response = await page.goto("/no-such-page");
    expect(response?.status()).toBe(404);
    await expect(page.getByRole("heading", { name: "No page here" })).toBeVisible();
  });
});
