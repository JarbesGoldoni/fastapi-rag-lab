import { describe, expect, it } from "vitest";
import { progressView } from "./progress";
import type { DocumentInfo } from "./types";

const doc = (patch: Partial<DocumentInfo>): DocumentInfo => ({
  id: "d",
  filename: "f.pdf",
  size_bytes: 1,
  status: "processing",
  pages: null,
  chunks: null,
  extractor: null,
  contextualized: null,
  warnings: [],
  error: null,
  progress: null,
  created_at: "",
  ...patch,
});

describe("progressView", () => {
  it("shows the stage, the real count and the server's percent", () => {
    const view = progressView(
      doc({ progress: { stage: "embedding", done: 128, total: 340, percent: 37 } }),
    );
    expect(view).toEqual({ label: "Embedding", count: "128 / 340 passages", percent: 37 });
  });

  it("formats large counts with separators", () => {
    const view = progressView(
      doc({ progress: { stage: "embedding", done: 448, total: 1426, percent: 31 } }),
    );
    expect(view?.count).toBe("448 / 1,426 passages");
  });

  it("is indeterminate when the total is unknown", () => {
    const view = progressView(doc({ progress: { stage: "reading", done: 0, total: 0, percent: 0 } }));
    expect(view?.percent).toBeNull();
  });

  it("covers queued and finished documents", () => {
    expect(progressView(doc({ status: "queued" }))?.label).toBe("Waiting in queue");
    expect(progressView(doc({ status: "ready" }))).toBeNull();
  });
});
