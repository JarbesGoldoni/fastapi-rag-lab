import { describe, expect, it } from "vitest";
import { chatReducer, historyOf, type Message } from "./chat";

describe("chatReducer", () => {
  it("streams an answer, resets on restart, and completes", () => {
    let state: Message[] = [];
    state = chatReducer(state, { type: "ask", id: "a1", question: "How long?", provider: "p" });
    state = chatReducer(state, { type: "delta", id: "a1", text: "Refunds are cre" });
    state = chatReducer(state, { type: "reset", id: "a1" });
    state = chatReducer(state, { type: "delta", id: "a1", text: "Thirty days [1]." });
    expect(state[1]!.text).toBe("Thirty days [1].");
    state = chatReducer(state, { type: "done", id: "a1", answer: "Thirty days [1].", citations: [] });
    expect(state.map((m) => [m.role, m.status])).toEqual([
      ["user", "done"],
      ["assistant", "done"],
    ]);
  });

  it("does not mutate the previous state", () => {
    const before: Message[] = [];
    const after = chatReducer(before, { type: "ask", id: "x", question: "q", provider: "p" });
    expect(before).toEqual([]);
    expect(after).toHaveLength(2);
  });

  it("history keeps only finished, non-empty turns", () => {
    let state: Message[] = [];
    state = chatReducer(state, { type: "ask", id: "a", question: "q1", provider: "p" });
    state = chatReducer(state, { type: "done", id: "a", answer: "a1", citations: [] });
    state = chatReducer(state, { type: "ask", id: "b", question: "q2", provider: "p" });
    expect(historyOf(state)).toEqual([
      { role: "user", text: "q1" },
      { role: "assistant", text: "a1" },
      { role: "user", text: "q2" },
    ]);
  });
});
