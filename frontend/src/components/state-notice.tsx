import { AlertTriangle, Inbox, RotateCw } from "lucide-react";
import { Button } from "@/components/ui/button";
import type { ApiError } from "@/lib/api/client";
import { cn } from "@/lib/utils";

/** The empty and error states every view needs, in one place, so they look and read the same. */
export function EmptyState({ title, children }: { title: string; children?: React.ReactNode }) {
  return (
    <div className="flex flex-col items-center gap-2 px-6 py-16 text-center">
      <Inbox className="size-6 text-muted-foreground" aria-hidden />
      <p className="font-display text-lg">{title}</p>
      {children && <p className="max-w-md text-sm text-muted-foreground">{children}</p>}
    </div>
  );
}

export function describeError(error: ApiError): string {
  if (error.unreachable) return "WikiGuard's API isn't answering. It may be waking up, which can take a minute on the free plan.";
  if (error.code === "rate_limited") return "Too many requests for a moment.";
  if (error.code === "database_unavailable") return "The database is unavailable for a moment.";
  return error.detail ?? error.message;
}

export function ErrorState({
  error,
  onRetry,
  className,
}: {
  error: ApiError;
  onRetry?: () => void;
  className?: string;
}) {
  return (
    <div role="alert" className={cn("flex flex-col items-center gap-3 px-6 py-14 text-center", className)}>
      <AlertTriangle className="size-6 text-risk-elevated" aria-hidden />
      <p className="font-display text-lg">Couldn’t load this</p>
      <p className="max-w-md text-sm text-muted-foreground">{describeError(error)}</p>
      {error.requestId && (
        <p className="font-mono text-[11px] text-muted-foreground">Request {error.requestId}</p>
      )}
      {onRetry && (
        <Button variant="outline" size="sm" onClick={onRetry}>
          <RotateCw aria-hidden /> Try again
        </Button>
      )}
    </div>
  );
}
