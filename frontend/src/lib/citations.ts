import type { Citation, Hit } from "./types";

/** Same rule as the backend: cited [n] markers, first-mention order, valid numbers only. */
export function citationsOf(answer: string, hits: Hit[]): Citation[] {
  const seen = new Set<number>();
  for (const match of answer.matchAll(/\[(\d{1,3})\]/g)) {
    const n = Number(match[1]);
    if (n >= 1 && n <= hits.length) seen.add(n);
  }
  return [...seen].map((n) => ({ number: n, hit: hits[n - 1]! }));
}

/** Split an answer into text and citation tokens for rendering. */
export type Segment = { kind: "text"; text: string } | { kind: "cite"; number: number };

export function segments(answer: string): Segment[] {
  const out: Segment[] = [];
  let last = 0;
  for (const match of answer.matchAll(/\[(\d{1,3})\]/g)) {
    const at = match.index ?? 0;
    if (at > last) out.push({ kind: "text", text: answer.slice(last, at) });
    out.push({ kind: "cite", number: Number(match[1]) });
    last = at + match[0].length;
  }
  if (last < answer.length) out.push({ kind: "text", text: answer.slice(last) });
  return out;
}
