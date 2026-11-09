"use client";

import { ErrorPanel } from "@/components/error-panel";

/** Catches a crash in any page outside the console, and keeps the rest of the site working. */
export default function RootError({
  error,
  retry,
}: {
  error: Error & { digest?: string };
  retry: () => void;
}) {
  return (
    <ErrorPanel
      code="500"
      title="Something broke on this page"
      body="The rest of WikiGuard should still work. Trying again often helps."
      digest={error.digest}
      onRetry={retry}
    />
  );
}
