import { cn } from "@/lib/utils";

/** A page of text with one line redlined: what a patroller does all day. */
export function LogoMark({ className }: { className?: string }) {
  return (
    <svg viewBox="0 0 24 24" aria-hidden className={cn("size-6", className)}>
      <rect x="3" y="2.5" width="18" height="19" rx="3.5" className="fill-card stroke-foreground" strokeWidth="1.5" />
      <path d="M7 8h10M7 12h6M7 16h8" className="stroke-muted-foreground" strokeWidth="1.5" strokeLinecap="round" />
      <path d="M5.5 12h13" className="stroke-risk-flagged" strokeWidth="2" strokeLinecap="round" />
    </svg>
  );
}

export function Wordmark({ className }: { className?: string }) {
  return (
    <span className={cn("inline-flex items-center gap-2", className)}>
      <LogoMark />
      <span className="font-display text-[19px] leading-none font-semibold tracking-[-0.01em]">
        Wiki<span className="italic font-medium">Guard</span>
      </span>
    </span>
  );
}
