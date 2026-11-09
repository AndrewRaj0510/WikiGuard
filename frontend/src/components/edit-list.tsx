"use client";

import { useCallback, useRef, useState } from "react";
import Link from "next/link";
import { useWindowVirtualizer } from "@tanstack/react-virtual";
import { EyeOff, Feather, LifeBuoy, UserRoundX } from "lucide-react";
import { Chip } from "@/components/ui/chip";
import { RISK_BAR, RiskBadge } from "@/components/risk";
import { Skeleton } from "@/components/ui/skeleton";
import type { EditSummary } from "@/lib/api/types";
import { absoluteTime, formatBytesDelta, relativeTime } from "@/lib/format";
import { riskBand } from "@/lib/risk";
import { useShortcuts } from "@/lib/shortcuts";
import { useNow } from "@/lib/use-now";
import { cn } from "@/lib/utils";

const DECISION_LABEL = {
  vandalism: "Reviewed: vandalism",
  good_faith_mistake: "Reviewed: good-faith mistake",
  ok: "Reviewed: OK",
  skip: "Skipped",
} as const;

const COLUMNS =
  "grid grid-cols-[minmax(0,1fr)_auto] gap-x-4 gap-y-1.5 sm:grid-cols-[7.5rem_minmax(0,1fr)_4.75rem_7.5rem] sm:items-center";

/** The column heads, for sighted users. Each row also reads well on its own to a screen reader. */
export function EditListHeader() {
  return (
    <div
      aria-hidden
      className={cn(
        COLUMNS,
        "hidden border-b px-4 py-2 text-[11px] font-medium tracking-[0.08em] text-muted-foreground uppercase sm:grid",
      )}
    >
      <span>Risk</span>
      <span>Article and summary</span>
      <span className="text-right">Change</span>
      <span className="text-right">Model · when</span>
    </div>
  );
}

/**
 * A list of edits. `virtual` renders only the rows near the screen (for the queue, which can grow
 * to thousands with "Load more"); `keyboard` adds j and k to move between rows (PLAN section 9).
 */
export function EditList({
  items,
  arrived,
  label,
  virtual = false,
  keyboard = false,
}: {
  items: EditSummary[];
  /** Rows that just came in, which get a brief highlight. */
  arrived?: ReadonlySet<number>;
  label: string;
  virtual?: boolean;
  keyboard?: boolean;
}) {
  if (virtual) return <VirtualList items={items} arrived={arrived} label={label} keyboard={keyboard} />;
  return <PlainList items={items} arrived={arrived} label={label} keyboard={keyboard} />;
}

interface ListProps {
  items: EditSummary[];
  arrived?: ReadonlySet<number>;
  label: string;
  keyboard: boolean;
}

/** Focuses row `index`'s link, waiting a few frames if the virtual list hasn't drawn it yet. */
function focusRow(list: HTMLElement | null, index: number, tries = 6): void {
  const link = list?.querySelector<HTMLAnchorElement>(`[data-index="${index}"] a`);
  if (link) {
    link.focus();
    return;
  }
  if (tries > 0) requestAnimationFrame(() => focusRow(list, index, tries - 1));
}

/** j and k move from the focused row, or start at the top. */
function useRowKeys(
  enabled: boolean,
  count: number,
  list: React.RefObject<HTMLOListElement | null>,
  reveal?: (index: number) => void,
) {
  useShortcuts(
    {
      j: () => move(1),
      k: () => move(-1),
    },
    enabled && count > 0,
  );
  function move(step: 1 | -1) {
    const focused = document.activeElement?.closest<HTMLElement>("[data-index]");
    const from = focused && list.current?.contains(focused) ? Number(focused.dataset.index) : -1;
    const next = Math.min(count - 1, Math.max(0, from + step));
    reveal?.(next);
    focusRow(list.current, next);
  }
}

function PlainList({ items, arrived, label, keyboard }: ListProps) {
  const list = useRef<HTMLOListElement>(null);
  useRowKeys(keyboard, items.length, list);
  return (
    <ol ref={list} aria-label={label}>
      {items.map((item, index) => (
        <EditRow key={item.rev_id} item={item} index={index} arrived={arrived?.has(item.rev_id) ?? false} />
      ))}
    </ol>
  );
}

const ESTIMATED_ROW_PX = 72;

function VirtualList({ items, arrived, label, keyboard }: ListProps) {
  // TanStack Virtual keeps its own mutable state, which the React Compiler can't memoize safely.
  "use no memo";
  const list = useRef<HTMLOListElement>(null);
  const [offset, setOffset] = useState(0);
  // Where the list starts on the page, so the window virtualizer knows which rows are on screen.
  const measure = useCallback((node: HTMLOListElement | null) => {
    list.current = node;
    if (node) setOffset(node.getBoundingClientRect().top + window.scrollY);
  }, []);
  const virtualizer = useWindowVirtualizer({
    count: items.length,
    estimateSize: () => ESTIMATED_ROW_PX,
    overscan: 10,
    scrollMargin: offset,
    getItemKey: (index) => items[index]?.rev_id ?? index,
  });
  useRowKeys(keyboard, items.length, list, (index) =>
    virtualizer.scrollToIndex(index, { align: "auto" }),
  );
  const rows = virtualizer.getVirtualItems();
  return (
    <ol
      ref={measure}
      aria-label={label}
      className="relative"
      style={{ height: virtualizer.getTotalSize() }}
    >
      {rows.map((row) => {
        const item = items[row.index];
        if (!item) return null;
        return (
          <EditRow
            key={row.key}
            item={item}
            index={row.index}
            total={items.length}
            arrived={arrived?.has(item.rev_id) ?? false}
            measureRef={virtualizer.measureElement}
            style={{ position: "absolute", top: 0, left: 0, width: "100%", transform: `translateY(${row.start - virtualizer.options.scrollMargin}px)` }}
          />
        );
      })}
    </ol>
  );
}

function EditRow({
  item,
  arrived,
  index,
  total,
  measureRef,
  style,
}: {
  item: EditSummary;
  arrived: boolean;
  index: number;
  /** Set in the virtual list, where the browser can't count the rows it doesn't draw. */
  total?: number;
  measureRef?: (node: HTMLLIElement | null) => void;
  style?: React.CSSProperties;
}) {
  const now = useNow();
  const band = riskBand(item.score, item.threshold);
  const delta = item.bytes_delta ?? 0;
  return (
    <li
      ref={measureRef}
      data-index={index}
      aria-setsize={total}
      aria-posinset={total ? index + 1 : undefined}
      style={style}
      className={cn(
        COLUMNS,
        "group relative border-b px-4 py-3 transition-colors last:border-b-0 hover:bg-accent/60 focus-within:bg-accent/60",
        arrived && "animate-arrive",
      )}
    >
      <span
        aria-hidden
        className={cn("absolute inset-y-2.5 left-0 w-[3px] rounded-r-full", RISK_BAR[band])}
      />
      <div className="col-span-2 sm:col-span-1">
        <RiskBadge score={item.score} threshold={item.threshold} />
      </div>

      <div className="col-span-2 min-w-0 sm:col-span-1">
        <h3 className="truncate font-display text-[16.5px] leading-snug font-medium">
          {/* The whole row is the link's hit area; the title is its accessible name. */}
          <Link
            href={`/edits/${item.rev_id}`}
            className="outline-none after:absolute after:inset-0 after:rounded-sm focus-visible:after:outline-2 focus-visible:after:outline-ring decoration-risk-flagged/60 underline-offset-[3px] group-hover:underline"
          >
            {item.title}
          </Link>
        </h3>
        <div className="mt-1 flex min-w-0 items-center gap-1.5">
          {item.is_anonymous && (
            <Chip tone="outline" title="Edited while logged out (an IP or a temporary account)">
              <UserRoundX aria-hidden /> Logged out
            </Chip>
          )}
          {item.is_minor && (
            <Chip tone="outline" title="Marked as a minor edit">
              <Feather aria-hidden /> Minor
            </Chip>
          )}
          {item.used_fallback && (
            <Chip tone="warn" title="The champion model didn't answer in time, so a fallback scored this edit">
              <LifeBuoy aria-hidden /> Fallback
            </Chip>
          )}
          {item.content_hidden && (
            <Chip tone="neutral">
              <EyeOff aria-hidden /> Hidden by Wikipedia
            </Chip>
          )}
          {item.decision && <Chip tone="ok">{DECISION_LABEL[item.decision]}</Chip>}
          <p className="min-w-0 truncate text-[13px] text-muted-foreground">
            {item.comment ? (
              item.comment
            ) : (
              <span className="italic">No edit summary</span>
            )}
          </p>
        </div>
      </div>

      <p
        className={cn(
          "font-mono text-[12.5px] tabular sm:text-right",
          delta > 0 && "text-diff-added",
          delta < 0 && "text-diff-removed",
          delta === 0 && "text-muted-foreground",
        )}
      >
        <span className="sr-only">Change in size: </span>
        {formatBytesDelta(item.bytes_delta)}
        <span className="sr-only"> bytes</span>
      </p>

      <p className="text-right text-[12px] leading-tight text-muted-foreground">
        <span className="block truncate font-mono text-[11.5px]">{item.model_name ?? "—"}</span>
        <time dateTime={item.edited_at} title={absoluteTime(item.edited_at)} className="tabular">
          {relativeTime(item.edited_at, now)}
        </time>
      </p>
    </li>
  );
}

export function EditListSkeleton({ rows = 8 }: { rows?: number }) {
  return (
    <div aria-busy="true" aria-label="Loading edits" className="divide-y">
      {Array.from({ length: rows }, (_, index) => (
        <div key={index} className={cn(COLUMNS, "px-4 py-3.5")}>
          <Skeleton className="h-4 w-20" />
          <div className="space-y-2">
            <Skeleton className="h-4 w-3/5" />
            <Skeleton className="h-3 w-2/5" />
          </div>
          <Skeleton className="hidden h-3.5 w-12 justify-self-end sm:block" />
          <Skeleton className="hidden h-3.5 w-16 justify-self-end sm:block" />
        </div>
      ))}
    </div>
  );
}
