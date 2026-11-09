import { EyeOff } from "lucide-react";
import { cn } from "@/lib/utils";

/**
 * Removed and added text, side by side. Vandals write this text, so it's only ever a React text
 * node: no HTML, no markdown, and no links are made from it (PLAN section 8).
 * Each line carries a − or + mark as well as a color, so it reads without color too.
 */
export function DiffView({
  removed,
  added,
  hidden,
}: {
  removed: string | null;
  added: string | null;
  hidden: boolean;
}) {
  if (hidden) {
    return (
      <div className="flex items-start gap-3 rounded-lg border bg-muted/60 p-5 text-sm text-muted-foreground">
        <EyeOff className="mt-0.5 size-4 shrink-0" aria-hidden />
        <p>
          Wikipedia has hidden this revision’s text, usually because it held personal
          information. WikiGuard deleted its copy and won’t show or explain it.
        </p>
      </div>
    );
  }
  return (
    <div className="grid gap-4 md:grid-cols-2">
      <Pane kind="removed" text={removed} />
      <Pane kind="added" text={added} />
    </div>
  );
}

const PANE = {
  removed: {
    title: "Removed",
    mark: "−",
    empty: "Nothing removed",
    tone: "text-diff-removed",
    soft: "bg-diff-removed-soft",
    rule: "border-diff-removed/30",
  },
  added: {
    title: "Added",
    mark: "+",
    empty: "Nothing added",
    tone: "text-diff-added",
    soft: "bg-diff-added-soft",
    rule: "border-diff-added/30",
  },
} as const;

function Pane({ kind, text }: { kind: keyof typeof PANE; text: string | null }) {
  const style = PANE[kind];
  const lines = text ? text.split("\n") : [];
  const characters = text?.length ?? 0;
  return (
    <section
      aria-label={`${style.title} text`}
      className={cn("min-w-0 overflow-hidden rounded-lg border bg-card", style.rule)}
    >
      <header className="flex items-center justify-between border-b px-4 py-2.5">
        <h3 className={cn("flex items-center gap-2 text-[13px] font-semibold", style.tone)}>
          <span
            aria-hidden
            className={cn("grid size-5 place-items-center rounded-[4px] font-mono text-sm", style.soft)}
          >
            {style.mark}
          </span>
          {style.title}
        </h3>
        <p className="font-mono text-[11.5px] text-muted-foreground tabular">
          {characters.toLocaleString("en-US")} chars
        </p>
      </header>
      {lines.length === 0 ? (
        <p className="px-4 py-6 text-sm text-muted-foreground italic">{style.empty}</p>
      ) : (
        <div className={cn("max-h-[34rem] overflow-auto py-2", style.soft)}>
          {lines.map((line, index) => (
            <div key={index} className="grid grid-cols-[2rem_minmax(0,1fr)] pr-4">
              <span
                aria-hidden
                className={cn("text-center font-mono text-[13px] leading-6 select-none", style.tone)}
              >
                {style.mark}
              </span>
              <span className="font-mono text-[13px] leading-6 break-words whitespace-pre-wrap">
                {line || " "}
              </span>
            </div>
          ))}
        </div>
      )}
    </section>
  );
}
