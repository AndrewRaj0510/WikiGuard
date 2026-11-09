"use client";

import { ErrorPanel } from "@/components/error-panel";

/** A crash in one console page keeps the header, the seat, and the other pages working. */
export default function ConsoleError({
  error,
  retry,
}: {
  error: Error & { digest?: string };
  retry: () => void;
}) {
  return (
    <ErrorPanel
      code="500"
      title="This view hit an error"
      body="Your session and seat are fine. Try again, or open another page from the menu."
      digest={error.digest}
      onRetry={retry}
    />
  );
}
