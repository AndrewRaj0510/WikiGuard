"use client";

import Link from "next/link";
import { RotateCw } from "lucide-react";
import { Button, buttonVariants } from "@/components/ui/button";

/**
 * What every error boundary shows: what happened, the digest to quote, and two ways on. The
 * error's message isn't shown: in production React replaces it anyway, and it could carry detail
 * that isn't meant for visitors.
 */
export function ErrorPanel({
  code,
  title,
  body,
  digest,
  onRetry,
}: {
  code: string;
  title: string;
  body: string;
  digest?: string;
  onRetry?: () => void;
}) {
  return (
    <div className="mx-auto flex min-h-[60vh] max-w-lg flex-col justify-center px-4 py-16">
      <p className="font-mono text-[13px] text-risk-flagged">{code}</p>
      <h1 className="mt-2 font-display text-[34px] leading-tight font-medium tracking-[-0.01em]">
        {title}
      </h1>
      <p className="mt-3 text-[15px] leading-relaxed text-muted-foreground">{body}</p>
      {digest && (
        <p className="mt-3 font-mono text-[11.5px] text-muted-foreground">Reference {digest}</p>
      )}
      <div className="mt-6 flex flex-wrap gap-3">
        {onRetry && (
          <Button onClick={onRetry}>
            <RotateCw aria-hidden /> Try again
          </Button>
        )}
        <Link href="/" className={buttonVariants({ variant: "outline" })}>
          Home
        </Link>
        <Link href="/queue" className={buttonVariants({ variant: "ghost" })}>
          Review queue
        </Link>
      </div>
    </div>
  );
}
