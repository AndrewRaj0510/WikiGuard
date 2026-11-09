import Link from "next/link";
import { EXTERNAL_REL } from "@/lib/wikipedia";

export const REPO_URL = "https://github.com/AndrewRaj0510/WikiGuard";
/** The owner chose GitHub Issues as the public contact for content reports (2026-11-09). */
export const REPORT_URL = `${REPO_URL}/issues/new?title=${encodeURIComponent("Content report")}`;

/** Attribution, privacy, and how to report content, on every page (PLAN section 8). */
export function SiteFooter() {
  return (
    <footer className="border-t">
      <div className="mx-auto flex max-w-6xl flex-col gap-3 px-4 py-6 text-[12.5px] text-muted-foreground sm:flex-row sm:items-start sm:justify-between">
        <p className="max-w-xl leading-relaxed">
          Edit text and titles come from{" "}
          <a href="https://en.wikipedia.org" rel={EXTERNAL_REL} className="underline underline-offset-2">
            English Wikipedia
          </a>
          , shared under{" "}
          <a
            href="https://creativecommons.org/licenses/by-sa/4.0/"
            rel={`license ${EXTERNAL_REL}`}
            className="underline underline-offset-2"
          >
            CC BY-SA 4.0
          </a>
          . Each edit links to its page on Wikipedia, which lists its authors. WikiGuard is an
          independent portfolio project, not affiliated with the Wikimedia Foundation.
        </p>
        <nav aria-label="About" className="flex shrink-0 flex-wrap gap-x-4 gap-y-1">
          <Link href="/privacy" className="underline-offset-2 hover:underline">
            Privacy
          </Link>
          <a href={REPORT_URL} rel={EXTERNAL_REL} className="underline-offset-2 hover:underline">
            Report content
          </a>
          <a href={REPO_URL} rel={EXTERNAL_REL} className="underline-offset-2 hover:underline">
            Source code
          </a>
        </nav>
      </div>
    </footer>
  );
}
