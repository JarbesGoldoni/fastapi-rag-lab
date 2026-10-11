import { parseSSE } from "./sse";
import type { AskEvent, DocumentInfo, Hit, Providers, Turn } from "./types";

async function json<T>(response: Response): Promise<T> {
  if (!response.ok) {
    const body = (await response.json().catch(() => ({}))) as { detail?: unknown };
    const detail = typeof body.detail === "string" ? body.detail : response.statusText;
    throw new Error(detail || `HTTP ${response.status}`);
  }
  return response.json() as Promise<T>;
}

export const getProviders = () => fetch("/api/providers").then(json<Providers>);

export const listDocuments = () => fetch("/api/documents").then(json<DocumentInfo[]>);

export interface UploadOptions {
  contextualizeWith?: string | null;
  aiExtraction?: string | null;
}

export function uploadDocument(file: Blob, name: string, options: UploadOptions) {
  const form = new FormData();
  form.append("file", file, name);
  if (options.contextualizeWith) form.append("contextualize_with", options.contextualizeWith);
  if (options.aiExtraction) form.append("ai_extraction", options.aiExtraction);
  return fetch("/api/documents", { method: "POST", body: form }).then(json<DocumentInfo>);
}

export async function deleteDocument(id: string): Promise<void> {
  const response = await fetch(`/api/documents/${id}`, { method: "DELETE" });
  if (!response.ok && response.status !== 404) throw new Error(`HTTP ${response.status}`);
}

export const fileUrl = (id: string, page?: number) =>
  `/api/documents/${id}/file${page ? `#page=${page}` : ""}`;

export interface AskBody {
  query: string;
  history: Turn[];
  document_ids: string[] | null;
}

const post = (path: string, body: unknown, signal?: AbortSignal) =>
  fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
    signal,
  });

export interface PromptParts {
  system: string;
  user: string;
}

/** For browser models: the prompt that turns a follow-up into a standalone search query. */
export const getRewritePrompt = (body: { query: string; history: Turn[] }) =>
  post("/api/rewrite-prompt", body).then(json<PromptParts>);

/** Sources and the answer prompt; `rewritten` is the browser model's raw rewrite, if any. */
export const getContext = (body: AskBody & { rewritten?: string }) =>
  post("/api/context", body).then(json<PromptParts & { hits: Hit[] }>);

/** POST /api/ask and deliver each server-sent event as it arrives. */
export async function askStream(
  body: AskBody & { provider: string },
  onEvent: (event: AskEvent) => void,
  signal: AbortSignal,
): Promise<void> {
  const response = await post("/api/ask", body, signal);
  if (!response.ok || !response.body) {
    await json(response);
    return;
  }
  const reader = response.body.pipeThrough(new TextDecoderStream()).getReader();
  let buffer = "";
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    const parsed = parseSSE(buffer + value);
    buffer = parsed.rest;
    parsed.events.forEach(onEvent);
  }
}
