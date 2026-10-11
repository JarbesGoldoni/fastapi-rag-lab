/** Messages between the UI thread and the voice workers. */

export type ToWorker =
  | { type: "load" }
  | { type: "transcribe"; id: number; audio: Float32Array }
  | { type: "speak"; id: number; text: string; voice: string };

export type FromWorker =
  | { type: "progress"; file: string; loaded: number; total: number }
  | { type: "ready" }
  | { type: "text"; id: number; text: string }
  | { type: "audio"; id: number; samples: Float32Array; sampleRate: number }
  | { type: "error"; id?: number; message: string };

/** Shape of transformers.js progress callbacks we care about. */
export interface ProgressLike {
  status: string;
  file?: string;
  loaded?: number;
  total?: number;
}

export function progressMessage(info: ProgressLike): FromWorker | null {
  if (info.status !== "progress" || !info.file) return null;
  return { type: "progress", file: info.file, loaded: info.loaded ?? 0, total: info.total ?? 0 };
}
