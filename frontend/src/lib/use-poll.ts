"use client";

import { useEffect, useRef, useState, useSyncExternalStore } from "react";
import { ApiError, isAbort } from "./api/client";

/**
 * Polls one resource while the tab is visible.
 *
 * - One request at a time: the next waits until the last one has answered (`setTimeout`, never
 *   `setInterval`), so a slow API can't pile requests up.
 * - Stops while the tab is hidden and polls at once on return. Every poll counts against free
 *   quotas (F-014), and nobody reads a hidden tab.
 * - Backs off after failures, doubling up to a minute, and honours `Retry-After`.
 * - Keeps the last good data through a failure, so a blip shows as "stale", not as an empty page.
 * - A new `key` starts fresh, so a changed filter never shows the old filter's rows.
 */
export interface PollOptions {
  intervalMs: number;
  enabled?: boolean;
}

export interface PollState<T> {
  data: T | undefined;
  error: ApiError | undefined;
  updatedAt: number | undefined;
  /** "loading" until the first answer for this key, then "ready", or "error" with no data at all. */
  status: "loading" | "ready" | "error";
  paused: boolean;
  refresh: () => void;
}

const MAX_BACKOFF_MS = 60_000;

export function backoffDelay(intervalMs: number, failures: number, retryAfterSeconds?: number) {
  const base = failures === 0 ? intervalMs : Math.min(intervalMs * 2 ** failures, MAX_BACKOFF_MS);
  return Math.max(base, (retryAfterSeconds ?? 0) * 1000);
}

function subscribeVisibility(notify: () => void): () => void {
  document.addEventListener("visibilitychange", notify);
  return () => document.removeEventListener("visibilitychange", notify);
}

export function usePageVisible(): boolean {
  return useSyncExternalStore(
    subscribeVisibility,
    () => document.visibilityState !== "hidden",
    () => true,
  );
}

function asApiError(error: unknown): ApiError {
  if (error instanceof ApiError) return error;
  return new ApiError({ status: 0, code: "client_error", title: "Something went wrong" });
}

interface Snapshot<T> {
  key: string;
  data: T | undefined;
  error: ApiError | undefined;
  updatedAt: number | undefined;
}

export function usePoll<T>(
  key: string,
  load: (signal: AbortSignal) => Promise<T>,
  { intervalMs, enabled = true }: PollOptions,
): PollState<T> {
  const visible = usePageVisible();
  const [snapshot, setSnapshot] = useState<Snapshot<T>>({
    key,
    data: undefined,
    error: undefined,
    updatedAt: undefined,
  });
  const loadRef = useRef(load);
  const runNow = useRef<() => void>(() => {});

  useEffect(() => {
    loadRef.current = load;
  });

  useEffect(() => {
    if (!enabled || !visible) return;
    let stopped = false;
    let failures = 0;
    let timer: ReturnType<typeof setTimeout> | undefined;
    let controller: AbortController | undefined;

    const run = async () => {
      clearTimeout(timer);
      controller?.abort();
      controller = new AbortController();
      let retryAfter: number | undefined;
      try {
        const data = await loadRef.current(controller.signal);
        if (stopped) return;
        failures = 0;
        setSnapshot({ key, data, error: undefined, updatedAt: Date.now() });
      } catch (caught) {
        if (stopped || isAbort(caught)) return;
        failures += 1;
        const error = asApiError(caught);
        retryAfter = error.retryAfterSeconds;
        setSnapshot((previous) =>
          previous.key === key
            ? { ...previous, error }
            : { key, data: undefined, error, updatedAt: undefined },
        );
      }
      if (!stopped) timer = setTimeout(run, backoffDelay(intervalMs, failures, retryAfter));
    };

    runNow.current = () => void run();
    void run();
    return () => {
      stopped = true;
      clearTimeout(timer);
      controller?.abort();
      runNow.current = () => {};
    };
  }, [key, intervalMs, enabled, visible]);

  // A snapshot for another key is from before the key changed: treat it as nothing yet.
  const current = snapshot.key === key ? snapshot : undefined;
  const data = current?.data;
  const error = current?.error;
  return {
    data,
    error,
    updatedAt: current?.updatedAt,
    status: data !== undefined ? "ready" : error ? "error" : "loading",
    paused: !visible,
    refresh: () => runNow.current(),
  };
}
