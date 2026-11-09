import type { Metadata } from "next";
import { ConsoleShell } from "@/components/console/console-shell";

// Vandal text is on these pages, so search engines are asked not to index or follow them.
export const metadata: Metadata = {
  robots: { index: false, follow: false },
};

export default function ConsoleLayout({ children }: LayoutProps<"/">) {
  return <ConsoleShell>{children}</ConsoleShell>;
}
