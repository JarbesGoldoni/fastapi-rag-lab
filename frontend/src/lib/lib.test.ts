import { describe, expect, it } from "vitest";
import { citationsOf, segments } from "./citations";
import { takeSentences, speakable } from "./sentences";
import { parseSSE } from "./sse";
import type { Hit } from "./types";

describe("parseSSE", () => {
  it("returns complete events and keeps the partial tail", () => {
    const buffer =
      'event: delta\ndata: {"text":"Hel"}\n\nevent: delta\ndata: {"text":"lo"}\n\nevent: do';
    const { events, rest } = parseSSE(buffer);
    expect(events).toEqual([
      { type: "delta", text: "Hel" },
      { type: "delta", text: "lo" },
    ]);
    expect(rest).toBe("event: do");
  });

  it("maps reset, done and error events", () => {
    const buffer =
      'event: reset\ndata: {}\n\nevent: done\ndata: {"answer":"a","citations":[]}\n\n' +
      'event: error\ndata: {"message":"boom"}\n\n';
    expect(parseSSE(buffer).events).toEqual([
      { type: "reset" },
      { type: "done", answer: "a", citations: [] },
      { type: "error", message: "boom" },
    ]);
  });
});

describe("takeSentences", () => {
  it("emits finished sentences and strips citations", () => {
    const { sentences, rest } = takeSentences("Refunds take 14 days [1]. Card refunds go back to");
    expect(sentences).toEqual(["Refunds take 14 days."]);
    expect(rest).toBe(" Card refunds go back to");
  });

  it("waits when the period is the last character (could be 3.5)", () => {
    expect(takeSentences("The rate is 3.").sentences).toEqual([]);
    expect(takeSentences("The rate is 3.", { flush: true }).sentences).toEqual(["The rate is 3."]);
  });

  it("merges very short fragments into the next sentence", () => {
    expect(takeSentences("Yes. The window is thirty days. More").sentences).toEqual([
      "Yes. The window is thirty days.",
    ]);
  });

  it("speakable removes markdown symbols", () => {
    expect(speakable("**Bold** `code` [2]")).toBe("Bold code");
  });
});

describe("citations", () => {
  const hit = (id: number): Hit => ({
    chunk_id: id,
    document_id: "d",
    filename: "f.pdf",
    page: id,
    text: "t",
    score: 0,
  });

  it("keeps first-mention order and ignores out-of-range numbers", () => {
    const cited = citationsOf("A [2] b [1][2] c [9]", [hit(1), hit(2)]);
    expect(cited.map((c) => c.number)).toEqual([2, 1]);
  });

  it("segments text and markers", () => {
    expect(segments("Yes [1].")).toEqual([
      { kind: "text", text: "Yes " },
      { kind: "cite", number: 1 },
      { kind: "text", text: "." },
    ]);
  });
});
