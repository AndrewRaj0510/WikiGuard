import { act, renderHook } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { ApiError } from "./api/client";
import { usePoll } from "./use-poll";

function setVisibility(state: "visible" | "hidden") {
  Object.defineProperty(document, "visibilityState", { value: state, configurable: true });
  document.dispatchEvent(new Event("visibilitychange"));
}

beforeEach(() => {
  vi.useFakeTimers();
  setVisibility("visible");
});
afterEach(() => vi.useRealTimers());

async function flush(ms = 0) {
  await act(async () => {
    await vi.advanceTimersByTimeAsync(ms);
  });
}

describe("usePoll", () => {
  it("loads at once, then once per interval, one request at a time", async () => {
    let calls = 0;
    const load = vi.fn(async () => ++calls);
    const { result } = renderHook(() => usePoll("k", load, { intervalMs: 3000 }));
    expect(result.current.status).toBe("loading");
    await flush();
    expect(result.current).toMatchObject({ status: "ready", data: 1 });
    await flush(3000);
    expect(result.current.data).toBe(2);
    expect(load).toHaveBeenCalledTimes(2);
  });

  it("stops while the tab is hidden, and polls at once on return", async () => {
    const load = vi.fn(async () => "x");
    const { result } = renderHook(() => usePoll("k", load, { intervalMs: 3000 }));
    await flush();
    act(() => setVisibility("hidden"));
    expect(result.current.paused).toBe(true);
    await flush(30_000);
    expect(load).toHaveBeenCalledTimes(1);
    act(() => setVisibility("visible"));
    await flush();
    expect(load).toHaveBeenCalledTimes(2);
    expect(result.current.paused).toBe(false);
  });

  it("keeps the last good data through a failure, and backs off", async () => {
    let fail = false;
    const load = vi.fn(async () => {
      if (fail) throw new ApiError({ status: 0, code: "api_unreachable", title: "down" });
      return "good";
    });
    const { result } = renderHook(() => usePoll("k", load, { intervalMs: 3000 }));
    await flush();
    fail = true;
    await flush(3000);
    expect(result.current).toMatchObject({ status: "ready", data: "good" });
    expect(result.current.error?.code).toBe("api_unreachable");
    // One failure: the next try waits 6 s, not 3.
    await flush(3000);
    expect(load).toHaveBeenCalledTimes(2);
    await flush(3000);
    expect(load).toHaveBeenCalledTimes(3);
  });

  it("starts fresh for a new key, never showing the old key's data", async () => {
    const { result, rerender } = renderHook(
      ({ key }) => usePoll(key, async () => key, { intervalMs: 3000 }),
      { initialProps: { key: "all" } },
    );
    await flush();
    expect(result.current.data).toBe("all");
    rerender({ key: "logged_out" });
    expect(result.current).toMatchObject({ status: "loading", data: undefined });
    await flush();
    expect(result.current.data).toBe("logged_out");
  });
});
