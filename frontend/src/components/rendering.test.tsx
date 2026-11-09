/**
 * Vandals write titles, comments, and diff text, so each must reach the page as text only.
 * Task 3 adds the full XSS corpus through the real UI in Playwright; this is the unit-level check.
 */
import { render, screen, within } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { DiffView } from "./diff-view";
import { EditList } from "./edit-list";
import { RiskBadge } from "./risk";
import type { EditSummary } from "@/lib/api/types";

const PAYLOADS = [
  "<script>window.__pwned = 1</script>",
  '<img src=x onerror="window.__pwned = 1">',
  "[[javascript:alert(1)|click]] **bold** <b>tag</b>",
];

function row(overrides: Partial<EditSummary> = {}): EditSummary {
  return {
    rev_id: 101,
    title: "Ada Lovelace",
    edited_at: "2026-11-08T12:00:00Z",
    is_minor: false,
    is_anonymous: false,
    comment: "fix",
    bytes_delta: -12,
    score: 0.42,
    model_name: "roberta",
    model_version: "1",
    threshold: 0.9,
    used_fallback: false,
    load_shed: false,
    decision: null,
    content_hidden: false,
    wikipedia_url: "https://en.wikipedia.org/w/index.php?diff=101",
    ...overrides,
  };
}

describe("untrusted text", () => {
  it.each(PAYLOADS)("shows a title and comment as text: %s", (payload) => {
    const { container } = render(
      <EditList items={[row({ title: payload, comment: payload })]} label="Edits" />,
    );
    expect(container.querySelector("script, img, b")).toBeNull();
    expect(screen.getByRole("link", { name: payload })).toBeInTheDocument();
    expect((window as unknown as { __pwned?: number }).__pwned).toBeUndefined();
  });

  it.each(PAYLOADS)("shows diff text as text: %s", (payload) => {
    const { container } = render(<DiffView removed={payload} added={payload} hidden={false} />);
    expect(container.querySelector("script, img, b")).toBeNull();
    expect(screen.getAllByText(payload)).toHaveLength(2);
  });

  it("links each row only to the app's own edit page", () => {
    render(<EditList items={[row({ wikipedia_url: "javascript:alert(1)" })]} label="Edits" />);
    expect(screen.getByRole("link")).toHaveAttribute("href", "/edits/101");
  });
});

describe("edit rows", () => {
  it("labels logged-out edits in words (DEC-053)", () => {
    render(<EditList items={[row({ is_anonymous: true }), row({ rev_id: 102 })]} label="Edits" />);
    const [first, second] = screen.getAllByRole("listitem");
    expect(within(first).getByText("Logged out")).toBeInTheDocument();
    expect(within(second).queryByText("Logged out")).toBeNull();
  });

  it("never shows risk by color alone", () => {
    render(<RiskBadge score={0.95} threshold={0.9} />);
    expect(screen.getByText("Flagged")).toBeInTheDocument();
    expect(screen.getByText("95%")).toBeInTheDocument();
  });

  it("says when Wikipedia hid the text, and shows none of it", () => {
    render(<DiffView removed="secret" added="secret" hidden />);
    expect(screen.queryByText("secret")).toBeNull();
    expect(screen.getByText(/hidden this revision/)).toBeInTheDocument();
  });
});
