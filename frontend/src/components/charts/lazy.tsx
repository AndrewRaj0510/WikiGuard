"use client";

import dynamic from "next/dynamic";
import { Skeleton } from "@/components/ui/skeleton";

/**
 * Charts load only when a page shows one, and only in the browser: Recharts is the largest
 * library in the app, and the data needs the tab's seat anyway (PLAN section 13).
 */
export const PrCurvesChart = dynamic(() => import("./pr-curves-chart"), {
  ssr: false,
  loading: () => <Skeleton className="h-[320px] w-full" />,
});

export const LiveMetricsChart = dynamic(() => import("./live-metrics-chart"), {
  ssr: false,
  loading: () => <Skeleton className="h-[240px] w-full" />,
});
