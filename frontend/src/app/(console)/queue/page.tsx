import type { Metadata } from "next";
import { Suspense } from "react";
import { QueueView } from "@/components/views/queue-view";
import { EditListSkeleton } from "@/components/edit-list";

export const metadata: Metadata = { title: "Review queue" };


export default function QueuePage() {
  // useSearchParams (the filter) suspends during prerendering, so the shell needs a fallback.
  return (
    <Suspense fallback={<EditListSkeleton />}>
      <QueueView />
    </Suspense>
  );
}
