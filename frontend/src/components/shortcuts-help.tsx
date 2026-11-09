"use client";

import { useRef } from "react";
import { Keyboard, X } from "lucide-react";
import { Button } from "@/components/ui/button";
import { SHORTCUTS, useShortcuts } from "@/lib/shortcuts";

/**
 * The shortcut list, in a native <dialog>: the browser traps focus, Escape closes it, and focus
 * returns to the button that opened it, with no extra code.
 */
export function ShortcutsHelp() {
  const dialog = useRef<HTMLDialogElement>(null);
  const open = () => dialog.current?.showModal();
  useShortcuts({ "?": open });
  return (
    <>
      <Button variant="ghost" size="icon" aria-label="Keyboard shortcuts" aria-keyshortcuts="?" onClick={open}>
        <Keyboard aria-hidden />
      </Button>
      <dialog
        ref={dialog}
        aria-labelledby="shortcuts-title"
        className="m-auto w-[min(32rem,calc(100vw-2rem))] rounded-lg border bg-card p-0 text-foreground shadow-2xl backdrop:bg-black/40"
      >
        <div className="flex items-center justify-between border-b px-5 py-3">
          <h2 id="shortcuts-title" className="font-display text-lg font-medium">
            Keyboard shortcuts
          </h2>
          <form method="dialog">
            <Button type="submit" variant="ghost" size="icon" aria-label="Close">
              <X aria-hidden />
            </Button>
          </form>
        </div>
        <table className="w-full text-left text-[13px]">
          <caption className="sr-only">Keyboard shortcuts</caption>
          <tbody className="divide-y">
            {SHORTCUTS.map((shortcut) => (
              <tr key={shortcut.keys}>
                <th scope="row" className="w-24 px-5 py-2 font-normal">
                  <kbd className="rounded border bg-muted px-1.5 py-0.5 font-mono text-[12px]">
                    {shortcut.keys}
                  </kbd>
                </th>
                <td className="py-2 pr-3">{shortcut.action}</td>
                <td className="py-2 pr-5 text-right text-[12px] text-muted-foreground">
                  {shortcut.where}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
        <p className="px-5 py-3 text-[12px] text-muted-foreground">
          Shortcuts pause while you type in a field, and each one is also a button on the page.
        </p>
      </dialog>
    </>
  );
}
