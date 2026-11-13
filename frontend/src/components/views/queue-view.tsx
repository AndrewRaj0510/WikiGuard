"use client";

import { useEffect, useState } from "react";
import { useSearchParams } from "next/navigation";
import { Button } from "@/components/ui/button";
import { Segmented } from "@/components/ui/segmented";
import { EditList, EditListHeader, EditListSkeleton } from "@/components/edit-list";
import { LiveStatus } from "@/components/live-status";
import { EmptyState, ErrorState, describeError } from "@/components/state-notice";
import { PageHeader } from "@/components/views/page-header";
import { ApiError } from "@/lib/api/client";
import type { EditorFilter, EditPage, EditSummary } from "@/lib/api/types";
import { setListOrder } from "@/lib/list-order";
import { useSessionStore } from "@/lib/session/provider";
import { usePoll } from "@/lib/use-poll";

export const QUEUE_PAGE_SIZE = 50;
// The queue changes slowly next to the feed: a new edit lands in it only if it ranks high.
const QUEUE_POLL_MS = 10_000;

const FILTERS: { value: EditorFilter; label: string }[] = [
  { value: "all", label: "All editors" },
  { value: "logged_out", label: "Logged out" },
  { value: "registered", label: "Registered" },
];

export function parseEditor(raw: string | null): EditorFilter {
  return raw === "logged_out" || raw === "registered" ? raw : "all";
}

/** Pages after the first, and the filter they belong to. */
interface Older {
  editor: EditorFilter;
  pages: EditPage[];
  loading: boolean;
  error: ApiError | null;
}

/** First page wins, then each older page adds only what's new: scores move between polls. */
export function mergePages(first: EditPage, older: EditPage[]): EditSummary[] {
  const seen = new Set<number>();
  const items: EditSummary[] = [];
  for (const page of [first, ...older]) {
    for (const item of page.items) {
      if (!seen.has(item.rev_id)) {
        seen.add(item.rev_id);
        items.push(item);
      }
    }
  }
  return items;
}

export function QueueView() {
  const store = useSessionStore();
  const editor = parseEditor(useSearchParams().get("editor"));
  const first = usePoll(
    `queue:${editor}`,
    (signal) => store.api.queue({ editor, limit: QUEUE_PAGE_SIZE }, signal),
    { intervalMs: QUEUE_POLL_MS },
  );
  const [olderState, setOlder] = useState<Older>({
    editor,
    pages: [],
    loading: false,
    error: null,
  });
  // Older pages from another filter don't belong here.
  const older = olderState.editor === editor ? olderState : null;

  function choose(next: EditorFilter) {
    // Shallow routing: the filter lives in the URL, so it survives a reload and can be shared,
    // without asking the server for a page it would render the same way.
    const url = next === "all" ? "/queue" : `/queue?editor=${next}`;
    window.history.replaceState(null, "", url);
  }

  async function loadMore() {
    const last = older?.pages.at(-1) ?? first.data;
    const cursor = last?.next_cursor;
    if (!cursor) return;
    const pages = older?.pages ?? [];
    setOlder({ editor, pages, loading: true, error: null });
    try {
      const page = await store.api.queue({ editor, limit: QUEUE_PAGE_SIZE, cursor });
      setOlder({ editor, pages: [...pages, page], loading: false, error: null });
    } catch (error) {
      setOlder({
        editor,
        pages,
        loading: false,
        error: error instanceof ApiError ? error : null,
      });
    }
  }

  const items = first.data ? mergePages(first.data, older?.pages ?? []) : [];
  const snapshot = first.data?.data_mode === "snapshot";
  const nextCursor = (older?.pages.at(-1) ?? first.data)?.next_cursor;
  const order = items.map((item) => item.rev_id).join(",");

  // So the edit page can step through this list in this order, and come back to this filter.
  useEffect(() => {
    setListOrder({
      ids: order ? order.split(",").map(Number) : [],
      href: editor === "all" ? "/queue" : `/queue?editor=${editor}`,
      label: "Review queue",
    });
  }, [order, editor]);

  return (
    <div className="mx-auto max-w-6xl px-4 py-8">
      <PageHeader
        eyebrow="Patrol"
        title="Review queue"
        lede={`Unreviewed edits ${snapshot ? "from the demo’s dataset" : "from the last three days"}, riskiest first. Logged-out edits rank with the rest and carry a label. WikiGuard never reverts anything: a person decides.`}
        aside={
          <LiveStatus
            paused={first.paused}
            error={first.error}
            updatedAt={first.updatedAt}
            serverStale={first.data?.stale}
            snapshot={snapshot}
          />
        }
      />

      <div className="mt-6 flex flex-wrap items-center justify-between gap-3">
        <Segmented
          name="editor"
          legend="Show edits by"
          options={FILTERS}
          value={editor}
          onChange={choose}
        />
        {first.data && (
          <p className="text-[12.5px] text-muted-foreground tabular">
            {items.length.toLocaleString("en-US")} shown{nextCursor ? ", more below" : ""}
          </p>
        )}
      </div>

      <div className="mt-4 overflow-hidden rounded-lg border bg-card">
        <EditListHeader />
        {first.status === "loading" && <EditListSkeleton />}
        {first.status === "error" && first.error && (
          <ErrorState error={first.error} onRetry={first.refresh} />
        )}
        {first.status === "ready" &&
          (items.length === 0 ? (
            <EmptyState title="The queue is clear">
              {editor === "all"
                ? `No unreviewed scored edits ${snapshot ? "in the dataset" : "from the last three days"}.`
                : "No unreviewed edits from this group. Try “All editors”."}
            </EmptyState>
          ) : (
            <EditList items={items} label="Review queue, riskiest first" virtual keyboard />
          ))}
      </div>

      {first.status === "ready" && nextCursor && (
        <div className="mt-4 flex flex-col items-center gap-2">
          <Button variant="outline" disabled={older?.loading} onClick={() => void loadMore()}>
            {older?.loading ? "Loading…" : "Load more"}
          </Button>
          {older?.error && (
            <p role="alert" className="text-sm text-risk-flagged">
              {describeError(older.error)}
            </p>
          )}
        </div>
      )}
    </div>
  );
}
