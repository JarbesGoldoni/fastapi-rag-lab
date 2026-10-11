export type DocumentStatus = "queued" | "processing" | "ready" | "failed";

export type Stage = "reading" | "context" | "embedding" | "saving";

export interface Progress {
  stage: Stage;
  done: number;
  /** 0 when the total is not known yet (a single AI call reading a scanned PDF). */
  total: number;
  percent: number;
}

export interface DocumentInfo {
  id: string;
  filename: string;
  size_bytes: number;
  status: DocumentStatus;
  pages: number | null;
  chunks: number | null;
  extractor: string | null;
  contextualized: string | null;
  warnings: string[];
  error: string | null;
  progress: Progress | null;
  created_at: string;
}

export interface Hit {
  chunk_id: number;
  document_id: string;
  filename: string;
  page: number;
  text: string;
  score: number;
}

export interface Citation {
  number: number;
  hit: Hit;
}

export interface Provider {
  id: string;
  label: string;
  kind: "cli" | "api" | "browser";
}

export interface Providers {
  generators: Provider[];
  transcribers: string[];
  embedding_model: string;
  default_generator: string | null;
}

export interface Turn {
  role: "user" | "assistant";
  text: string;
}

export type AskEvent =
  | { type: "sources"; hits: Hit[] }
  | { type: "delta"; text: string }
  | { type: "reset" }
  | { type: "done"; answer: string; citations: Citation[] }
  | { type: "error"; message: string };
