"use client";

import { useEffect, useRef } from "react";
import { EditList, EditListHeader, EditListSkeleton } from "@/components/edit-list";
import { LiveStatus } from "@/components/live-status";
import { EmptyState, ErrorState } from "@/components/state-notice";
import { PageHeader } from "@/components/views/page-header";
import type { EditPage } from "@/lib/api/types";
import { setListOrder } from "@/lib/list-order";
import { useSessionStore } from "@/lib/session/provider";
import { usePoll } from "@/lib/use-poll";

// PLAN section 9: every 3 s. Task 9 measures what this costs the free quotas (F-014).
export const FEED_POLL_MS = 3_000;
const FEED_SIZE = 50;

interface FeedSnapshot {
  page: EditPage;
  arrived: ReadonlySet<number>;
}

/** Which rows weren't in the last answer. The first answer marks none, so a load doesn't flash. */
export function newArrivals(previous: ReadonlySet<number> | null, page: EditPage): Set<number> {
  if (previous === null) return new Set();
  return new Set(page.items.filter((item) => !previous.has(item.rev_id)).map((i) => i.rev_id));
}

export function FeedView() {
  const store = useSessionStore();
  const seen = useRef<ReadonlySet<number> | null>(null);
  const feed = usePoll<FeedSnapshot>(
    "feed",
    async (signal) => {
      const page = await store.api.feed({ limit: FEED_SIZE }, signal);
      const arrived = newArrivals(seen.current, page);
      seen.current = new Set(page.items.map((item) => item.rev_id));
      return { page, arrived };
    },
    { intervalMs: FEED_POLL_MS },
  );
  const items = feed.data?.page.items ?? [];
  const snapshot = feed.data?.page.data_mode === "snapshot";
  const order = items.map((item) => item.rev_id).join(",");
  useEffect(() => {
    setListOrder({
      ids: order ? order.split(",").map(Number) : [],
      href: "/feed",
      label: "Edit feed",
    });
  }, [order]);

  return (
    <div className="mx-auto max-w-6xl px-4 py-8">
      <PageHeader
        eyebrow="Stream"
        title="Edit feed"
        lede={
          snapshot
            ? "Real English Wikipedia article edits from the demo’s dataset, newest first, each scored by the champion model. Bots are left out."
            : "English Wikipedia article edits as they arrive, newest first, each scored by the champion model or a fallback. Bots are left out."
        }
        aside={
          <LiveStatus
            paused={feed.paused}
            error={feed.error}
            updatedAt={feed.updatedAt}
            serverStale={feed.data?.page.stale}
            snapshot={snapshot}
          />
        }
      />
      <div className="mt-6 overflow-hidden rounded-lg border bg-card">
        <EditListHeader />
        {feed.status === "loading" && <EditListSkeleton />}
        {feed.status === "error" && feed.error && (
          <ErrorState error={feed.error} onRetry={feed.refresh} />
        )}
        {feed.status === "ready" &&
          (items.length === 0 ? (
            <EmptyState title={snapshot ? "No edits loaded" : "Waiting for edits"}>
              {snapshot
                ? "The demo’s dataset hasn’t been loaded yet."
                : "Nothing has arrived in the last three days. The ingestor may be stopped."}
            </EmptyState>
          ) : (
            <EditList
              items={items}
              arrived={feed.data?.arrived}
              label="Edit feed, newest first"
              keyboard
            />
          ))}
      </div>
    </div>
  );
}
