import type { Citation, Hit, Turn } from "./types";

export interface Message {
  id: string;
  role: "user" | "assistant";
  text: string;
  status: "streaming" | "done" | "error";
  provider?: string;
  hits?: Hit[];
  citations?: Citation[];
  error?: string;
}

export type ChatAction =
  | { type: "ask"; id: string; question: string; provider: string }
  | { type: "sources"; id: string; hits: Hit[] }
  | { type: "delta"; id: string; text: string }
  | { type: "reset"; id: string }
  | { type: "done"; id: string; answer: string; citations: Citation[] }
  | { type: "error"; id: string; message: string }
  | { type: "stop"; id: string }
  | { type: "clear" };

const update = (messages: Message[], id: string, change: (m: Message) => Message) =>
  messages.map((m) => (m.id === id ? change(m) : m));

/** Pure state transitions for the conversation. */
export function chatReducer(messages: Message[], action: ChatAction): Message[] {
  switch (action.type) {
    case "ask":
      return [
        ...messages,
        { id: `${action.id}-q`, role: "user", text: action.question, status: "done" },
        { id: action.id, role: "assistant", text: "", status: "streaming", provider: action.provider },
      ];
    case "sources":
      return update(messages, action.id, (m) => ({ ...m, hits: action.hits }));
    case "delta":
      return update(messages, action.id, (m) => ({ ...m, text: m.text + action.text }));
    case "reset":
      return update(messages, action.id, (m) => ({ ...m, text: "" }));
    case "done":
      return update(messages, action.id, (m) => ({
        ...m,
        text: action.answer,
        citations: action.citations,
        status: "done",
      }));
    case "error":
      return update(messages, action.id, (m) => ({ ...m, status: "error", error: action.message }));
    case "stop":
      return update(messages, action.id, (m) =>
        m.status === "streaming" ? { ...m, status: "done" } : m,
      );
    case "clear":
      return [];
  }
}

/** The finished turns sent back as conversation history. */
export function historyOf(messages: Message[], maxTurns = 6): Turn[] {
  return messages
    .filter((m) => m.status === "done" && m.text)
    .slice(-maxTurns)
    .map((m) => ({ role: m.role, text: m.text }));
}
