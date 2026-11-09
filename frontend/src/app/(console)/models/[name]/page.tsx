import type { Metadata } from "next";
import { Suspense } from "react";
import { Skeleton } from "@/components/ui/skeleton";
import { ModelCardView } from "@/components/views/model-card-view";

export const metadata: Metadata = { title: "Model card" };

// The name is read in the browser (useParams), like the edit page's rev ID.
export default function ModelCardPage() {
  return (
    <Suspense fallback={<Skeleton className="mx-auto mt-8 h-96 max-w-6xl" />}>
      <ModelCardView />
    </Suspense>
  );
}
