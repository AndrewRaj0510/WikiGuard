"use client";

import { useEffect, useRef } from "react";

/**
 * Single-key shortcuts for patrolling at speed (PLAN section 9: a keyboard-only review flow).
 *
 * A key never fires while someone types in a field, with a modifier held (so browser and screen
 * reader shortcuts keep working), or while a dialog is open. Each shortcut is also a visible
 * button, so none of them is the only way to do something (WCAG 2.1.4).
 */
export type Bindings = Record<string, (event: KeyboardEvent) => void>;

export function isTyping(target: EventTarget | null): boolean {
  if (!(target instanceof HTMLElement)) return false;
  if (target.isContentEditable) return true;
  return ["INPUT", "TEXTAREA", "SELECT"].includes(target.tagName);
}

export function shortcutKey(event: KeyboardEvent): string | null {
  if (event.defaultPrevented || event.ctrlKey || event.metaKey || event.altKey) return null;
  if (isTyping(event.target)) return null;
  if (typeof document !== "undefined" && document.querySelector("dialog[open]")) return null;
  return event.key.length === 1 ? event.key.toLowerCase() : event.key;
}

export function useShortcuts(bindings: Bindings, enabled = true): void {
  const latest = useRef(bindings);
  useEffect(() => {
    latest.current = bindings;
  });
  useEffect(() => {
    if (!enabled) return;
    const onKeyDown = (event: KeyboardEvent) => {
      // Shift isn't treated as a modifier, so "?" (Shift + / on most layouts) still arrives.
      const key = shortcutKey(event);
      if (key === null) return;
      const handler = latest.current[key];
      if (!handler) return;
      event.preventDefault();
      handler(event);
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [enabled]);
}

export const SHORTCUTS: { keys: string; action: string; where: string }[] = [
  { keys: "j / k", action: "Next or previous edit in the list", where: "Queue, feed" },
  { keys: "Enter", action: "Open the edit", where: "Queue, feed" },
  { keys: "v", action: "Mark as vandalism", where: "Edit (owner)" },
  { keys: "g", action: "Mark as a good-faith mistake", where: "Edit (owner)" },
  { keys: "o", action: "Mark as OK", where: "Edit (owner)" },
  { keys: "s", action: "Skip", where: "Edit (owner)" },
  { keys: "n / p", action: "Next or previous edit from the list you came from", where: "Edit" },
  { keys: "e", action: "Explain the edit", where: "Edit" },
  { keys: "b", action: "Back to the list", where: "Edit" },
  { keys: "?", action: "Show these shortcuts", where: "Anywhere" },
];
