import type { Metadata } from "next";
import Link from "next/link";
import { Wordmark } from "@/components/logo";
import { buttonVariants } from "@/components/ui/button";

export const metadata: Metadata = { title: "Not found", robots: { index: false } };

/** The custom 404. A server component, so it costs no JavaScript beyond the framework. */
export default function NotFound() {
  return (
    <div className="mx-auto flex min-h-dvh max-w-lg flex-col justify-center px-4 py-16">
      <Link href="/" aria-label="WikiGuard home" className="mb-10 self-start">
        <Wordmark />
      </Link>
      <p className="font-mono text-[13px] text-risk-flagged">404</p>
      <h1 className="mt-2 font-display text-[34px] leading-tight font-medium">
        No page here
      </h1>
      <p className="mt-3 text-[15px] leading-relaxed text-muted-foreground">
        The address may be mistyped, or the page has moved. Edits are kept for three days, so an
        older edit’s page is gone on purpose.
      </p>
      <div className="mt-6 flex gap-3">
        <Link href="/" className={buttonVariants()}>
          Home
        </Link>
        <Link href="/queue" className={buttonVariants({ variant: "outline" })}>
          Review queue
        </Link>
      </div>
    </div>
  );
}
