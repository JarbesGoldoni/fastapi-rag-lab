import type { DocumentInfo, Progress } from "./types";

const STAGES: Record<Progress["stage"], { label: string; unit: string }> = {
  reading: { label: "Reading", unit: "pages" },
  context: { label: "Adding context", unit: "passages" },
  embedding: { label: "Embedding", unit: "passages" },
  saving: { label: "Saving", unit: "passages" },
};

export interface ProgressView {
  label: string;
  /** e.g. "128 / 340 passages"; empty when the total is unknown. */
  count: string;
  /** null means indeterminate: show motion, not a number. */
  percent: number | null;
}

/** What the document row shows while it's being ingested. Every number comes from the server. */
export function progressView(doc: DocumentInfo): ProgressView | null {
  if (doc.status === "queued") return { label: "Waiting in queue", count: "", percent: null };
  if (doc.status !== "processing") return null;
  const progress = doc.progress;
  if (!progress) return { label: "Starting", count: "", percent: null };
  const stage = STAGES[progress.stage];
  if (progress.total === 0) return { label: stage.label, count: "", percent: null };
  return {
    label: stage.label,
    count: `${progress.done.toLocaleString("en-US")} / ${progress.total.toLocaleString("en-US")} ${stage.unit}`,
    percent: progress.percent,
  };
}
