import type { Metadata } from "next";
import { FeedView } from "@/components/views/feed-view";

export const metadata: Metadata = { title: "Live feed" };


export default function FeedPage() {
  return <FeedView />;
}
