/**
 * Chrome's built-in model (Gemini Nano) through the Prompt API. Runs fully in the browser.
 * Availability depends on the Chrome version and the machine; see README.
 */

type Availability = "unavailable" | "downloadable" | "downloading" | "available";

interface NanoSession {
  promptStreaming(input: string, options?: { signal?: AbortSignal }): ReadableStream<string>;
  destroy(): void;
}

interface NanoMonitor {
  addEventListener(type: "downloadprogress", listener: (event: { loaded: number }) => void): void;
}

interface LanguageModelApi {
  availability(options?: object): Promise<Availability>;
  create(options: {
    initialPrompts?: { role: "system" | "user" | "assistant"; content: string }[];
    expectedInputs?: { type: "text"; languages: string[] }[];
    expectedOutputs?: { type: "text"; languages: string[] }[];
    monitor?: (monitor: NanoMonitor) => void;
    signal?: AbortSignal;
  }): Promise<NanoSession>;
}

const api = (): LanguageModelApi | undefined =>
  (globalThis as { LanguageModel?: LanguageModelApi }).LanguageModel;

const LANGUAGES = { expectedInputs: [{ type: "text" as const, languages: ["en"] }], expectedOutputs: [{ type: "text" as const, languages: ["en"] }] };

export async function nanoAvailability(): Promise<Availability> {
  const model = api();
  if (!model) return "unavailable";
  try {
    return await model.availability(LANGUAGES);
  } catch {
    return "unavailable";
  }
}

/** Stream an answer; yields text deltas (older Chrome versions stream the cumulative text). */
export async function* nanoStream(
  system: string,
  user: string,
  signal: AbortSignal,
  onDownload: (fraction: number) => void,
): AsyncGenerator<string> {
  const model = api();
  if (!model) throw new Error("Chrome's built-in model is not available in this browser.");
  const session = await model.create({
    ...LANGUAGES,
    initialPrompts: [{ role: "system", content: system }],
    monitor: (m) => m.addEventListener("downloadprogress", (e) => onDownload(e.loaded)),
    signal,
  });
  try {
    const reader = session.promptStreaming(user, { signal }).getReader();
    let total = "";
    for (;;) {
      const { done, value } = await reader.read();
      if (done) return;
      const delta = value.startsWith(total) && total ? value.slice(total.length) : value;
      total += delta;
      if (delta) yield delta;
    }
  } finally {
    session.destroy();
  }
}
