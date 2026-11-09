import { cva, type VariantProps } from "class-variance-authority";
import { cn } from "@/lib/utils";

/** A small label. Every chip carries a word, so none depends on its color (WCAG 1.4.1). */
const chipVariants = cva(
  "inline-flex items-center gap-1 rounded-[5px] border px-1.5 py-px text-[11px] leading-4 font-medium tracking-wide whitespace-nowrap [&_svg]:size-3 [&_svg]:shrink-0",
  {
    variants: {
      tone: {
        neutral: "border-border bg-muted text-muted-foreground",
        outline: "border-border text-muted-foreground",
        warn: "border-risk-elevated/35 bg-risk-elevated-soft text-risk-elevated",
        danger: "border-risk-flagged/35 bg-risk-flagged-soft text-risk-flagged",
        ok: "border-diff-added/35 bg-diff-added-soft text-diff-added",
      },
    },
    defaultVariants: { tone: "neutral" },
  },
);

export type ChipProps = React.ComponentProps<"span"> & VariantProps<typeof chipVariants>;

export function Chip({ className, tone, ...props }: ChipProps) {
  return <span className={cn(chipVariants({ tone }), className)} {...props} />;
}
