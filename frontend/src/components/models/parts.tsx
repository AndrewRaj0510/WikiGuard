import { modelLabel, seriesColor } from "@/lib/models";
import { cn } from "@/lib/utils";

/** Small pieces shared by the Models, model card, and Health pages. */

export function Legend({ names, highlight }: { names: string[]; highlight: string | null }) {
  return (
    <ul className="mb-2 flex flex-wrap gap-x-4 gap-y-1 text-[12.5px]" aria-label="Legend">
      {names.map((name) => (
        <li key={name} className="flex items-center gap-1.5">
          <span
            aria-hidden
            className={cn("inline-block w-4 rounded-full", name === highlight ? "h-[3px]" : "h-0.5")}
            style={{ background: seriesColor(name) }}
          />
          <span className={name === highlight ? "font-semibold" : undefined}>
            {modelLabel(name)}
            {name === highlight ? " (champion)" : ""}
          </span>
        </li>
      ))}
    </ul>
  );
}

const TABLE_RECALLS = [10, 25, 50, 75];

/** The chart as a table, for screen readers, low contrast, and exact numbers. */
export function CurveTable({ series }: { series: { name: string; precision: number[] }[] }) {
  return (
    <details className="mt-3 text-[13px]">
      <summary className="cursor-pointer text-muted-foreground hover:text-foreground">
        View as a table
      </summary>
      <table className="mt-2 w-full max-w-xl text-left">
        <caption className="sr-only">Precision at four levels of recall</caption>
        <thead className="text-[11px] tracking-[0.08em] text-muted-foreground uppercase">
          <tr>
            <th scope="col" className="py-1.5 font-medium">Model</th>
            {TABLE_RECALLS.map((recall) => (
              <th key={recall} scope="col" className="py-1.5 text-right font-medium">
                at {recall}% recall
              </th>
            ))}
          </tr>
        </thead>
        <tbody className="divide-y">
          {series.map((item) => (
            <tr key={item.name}>
              <th scope="row" className="py-1.5 font-normal">{modelLabel(item.name)}</th>
              {TABLE_RECALLS.map((recall) => {
                const value = item.precision[recall - 1];
                return (
                  <td key={recall} className="py-1.5 text-right font-mono tabular">
                    {value !== undefined ? `${Math.round(value * 100)}%` : "—"}
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </details>
  );
}

export function NotMeasured() {
  return <span className="text-muted-foreground italic">not measured</span>;
}
