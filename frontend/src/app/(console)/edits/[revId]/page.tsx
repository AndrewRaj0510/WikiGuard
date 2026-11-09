import type { Metadata } from "next";
import { Suspense } from "react";
import { EditSkeleton, EditView } from "@/components/views/edit-view";

export const metadata: Metadata = { title: "Edit" };

// The rev ID is read in the browser (useParams), and the data needs the seat, which only the
// tab holds, so the server renders the same page for every ID and never reads params.
export default function EditPage() {
  return (
    <Suspense fallback={<EditSkeleton />}>
      <EditView />
    </Suspense>
  );
}
