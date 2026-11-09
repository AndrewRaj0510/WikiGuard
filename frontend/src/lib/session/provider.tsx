"use client";

import { createContext, use, useEffect, useState, useSyncExternalStore } from "react";
import { SessionStore, type SeatedState, type SessionState } from "./store";

const SessionContext = createContext<SessionStore | null>(null);
const STARTING: SessionState = { status: "starting", since: 0 };

/** One store per page load, which is what gives each tab its own in-memory tab ID. */
export function SessionProvider({ children }: { children: React.ReactNode }) {
  const [store] = useState(() => new SessionStore());
  useEffect(() => store.connect(window), [store]);
  return <SessionContext value={store}>{children}</SessionContext>;
}

export function useSessionStore(): SessionStore {
  const store = use(SessionContext);
  if (!store) throw new Error("useSessionStore needs a <SessionProvider>");
  return store;
}

export function useSession(): SessionState {
  const store = useSessionStore();
  // The server always renders "starting": a session only exists in the browser.
  return useSyncExternalStore(store.subscribe, store.getSnapshot, () => STARTING);
}

/** For views shown only once seated. The console shell guarantees that before rendering them. */
export function useSeat(): SeatedState {
  const state = useSession();
  if (state.status !== "seated") throw new Error("useSeat used outside a seated view");
  return state;
}
