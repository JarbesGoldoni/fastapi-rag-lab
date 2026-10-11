import type { AskEvent, Citation, Hit } from "./types";

export interface Parsed {
  events: AskEvent[];
  rest: string;
}

/** Split a text/event-stream buffer into complete events and the unfinished remainder. */
export function parseSSE(buffer: string): Parsed {
  const blocks = buffer.replace(/\r\n/g, "\n").split("\n\n");
  const rest = blocks.pop() ?? "";
  const events = blocks.flatMap((block) => {
    const fields = new Map<string, string>();
    for (const line of block.split("\n")) {
      const at = line.indexOf(":");
      if (at > 0) fields.set(line.slice(0, at), line.slice(at + 1).trimStart());
    }
    const name = fields.get("event");
    const data = fields.get("data");
    return name && data !== undefined ? [toEvent(name, JSON.parse(data))] : [];
  });
  return { events, rest };
}

function toEvent(name: string, data: unknown): AskEvent {
  switch (name) {
    case "sources":
      return { type: "sources", hits: data as Hit[] };
    case "delta":
      return { type: "delta", text: (data as { text: string }).text };
    case "reset":
      return { type: "reset" };
    case "done": {
      const d = data as { answer: string; citations: Citation[] };
      return { type: "done", answer: d.answer, citations: d.citations };
    }
    default:
      return { type: "error", message: (data as { message?: string }).message ?? "Unknown error" };
  }
}
