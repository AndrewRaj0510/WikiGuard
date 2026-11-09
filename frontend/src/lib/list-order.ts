/**
 * The order of the list a patroller last looked at, so the edit page can offer "next" and
 * "previous" in that order and a back link to that list. In memory only: it belongs to this tab
 * and this visit, like the seat.
 */
export interface ListOrder {
  ids: number[];
  /** Where "back" goes, filter included, such as "/queue?editor=logged_out". */
  href: string;
  label: string;
}

let current: ListOrder = { ids: [], href: "/queue", label: "Review queue" };

export function setListOrder(order: ListOrder): void {
  current = order;
}

export function listOrder(): ListOrder {
  return current;
}

/** The edit before or after this one in the last list, or null at either end or if it's not there. */
export function neighbour(revId: number, step: 1 | -1, order: ListOrder = current): number | null {
  const index = order.ids.indexOf(revId);
  if (index === -1) return null;
  return order.ids[index + step] ?? null;
}
