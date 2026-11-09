"use client";

import "./globals.css";

/**
 * The last resort: the root layout itself failed, so this page replaces it and brings its own
 * <html>, <body>, and styles. It uses system fonts, because the layout that loads ours is the
 * part that broke.
 */
export default function GlobalError({
  error,
  retry,
}: {
  error: Error & { digest?: string };
  retry: () => void;
}) {
  return (
    <html lang="en">
      <body className="min-h-dvh">
        <main className="mx-auto flex min-h-dvh max-w-lg flex-col justify-center px-4 py-16">
          <p className="font-mono text-[13px] text-risk-flagged">500</p>
          <h1 className="mt-2 text-[32px] leading-tight font-semibold">WikiGuard hit an error</h1>
          <p className="mt-3 text-[15px] leading-relaxed text-muted-foreground">
            The page couldn’t load. It’s usually brief: try again in a moment.
          </p>
          {error.digest && (
            <p className="mt-3 font-mono text-[11.5px] text-muted-foreground">
              Reference {error.digest}
            </p>
          )}
          <div className="mt-6 flex gap-3">
            <button
              type="button"
              onClick={() => retry()}
              className="h-9 rounded-md bg-primary px-4 text-sm font-medium text-primary-foreground"
            >
              Try again
            </button>
            {/* A full page load on purpose: client navigation may be what broke. */}
            {/* eslint-disable-next-line @next/next/no-html-link-for-pages */}
            <a href="/" className="h-9 rounded-md border px-4 text-sm leading-9 font-medium">
              Home
            </a>
          </div>
        </main>
      </body>
    </html>
  );
}
