/**
 * F-158: during a stream outage the feed's pill said "Live" while the banner said "Stale".
 * The pill now reads the newest edit's time as well as whether the API answered.
 */
import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { LiveStatus } from "./live-status";

const minutesAgo = (minutes: number) => new Date(Date.now() - minutes * 60_000).toISOString();

describe("LiveStatus", () => {
  it("says Live while the newest edit is recent", () => {
    render(
      <LiveStatus paused={false} error={undefined} updatedAt={Date.now()} newestAt={minutesAgo(1)} />,
    );
    expect(screen.getByText("Live")).toBeTruthy();
  });

  it("says Delayed once no edit has arrived for 10 minutes", () => {
    render(
      <LiveStatus paused={false} error={undefined} updatedAt={Date.now()} newestAt={minutesAgo(25)} />,
    );
    expect(screen.getByText("Delayed")).toBeTruthy();
    expect(screen.getByText(/newest edit 25m ago/)).toBeTruthy();
  });

  it("never calls the snapshot delayed: its edits are old by design", () => {
    render(
      <LiveStatus
        paused={false}
        error={undefined}
        updatedAt={Date.now()}
        newestAt={minutesAgo(60 * 24 * 20)}
        snapshot
      />,
    );
    expect(screen.getByText("Snapshot, not live")).toBeTruthy();
  });
});
