export function PageHeader({
  eyebrow,
  title,
  lede,
  aside,
}: {
  eyebrow: string;
  title: string;
  lede: string;
  aside?: React.ReactNode;
}) {
  return (
    <header className="flex animate-rise flex-col gap-4 sm:flex-row sm:items-end sm:justify-between">
      <div className="max-w-2xl">
        <p className="text-[11px] font-medium tracking-[0.14em] text-risk-flagged uppercase">
          {eyebrow}
        </p>
        <h1 className="mt-1.5 font-display text-[34px] leading-[1.1] font-medium tracking-[-0.015em]">
          {title}
        </h1>
        <p className="mt-2.5 text-[14.5px] leading-relaxed text-muted-foreground">{lede}</p>
      </div>
      {aside && <div className="shrink-0">{aside}</div>}
    </header>
  );
}
