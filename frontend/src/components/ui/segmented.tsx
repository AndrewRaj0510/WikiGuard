"use client";

import { cn } from "@/lib/utils";

export interface SegmentedOption<T extends string> {
  value: T;
  label: string;
}

/**
 * A choice of one, built on real radio inputs: arrow keys, focus, and screen readers work with
 * no extra code, and the selected option is never shown by color alone.
 */
export function Segmented<T extends string>({
  name,
  legend,
  options,
  value,
  onChange,
  className,
}: {
  name: string;
  legend: string;
  options: SegmentedOption<T>[];
  value: T;
  onChange: (value: T) => void;
  className?: string;
}) {
  return (
    <fieldset className={cn("inline-flex", className)}>
      <legend className="sr-only">{legend}</legend>
      <div className="inline-flex rounded-md border bg-muted p-0.5">
        {options.map((option) => (
          <label
            key={option.value}
            className={cn(
              "relative cursor-pointer rounded-[5px] px-3 py-1 text-[13px] font-medium text-muted-foreground transition-colors select-none",
              "has-[:focus-visible]:outline-2 has-[:focus-visible]:outline-offset-1 has-[:focus-visible]:outline-ring",
              "has-[:checked]:bg-card has-[:checked]:text-foreground has-[:checked]:shadow-[0_1px_2px_0_oklch(0_0_0/0.12)]",
              "hover:text-foreground",
            )}
          >
            <input
              type="radio"
              className="sr-only"
              name={name}
              value={option.value}
              checked={value === option.value}
              onChange={() => onChange(option.value)}
            />
            {option.label}
          </label>
        ))}
      </div>
    </fieldset>
  );
}
