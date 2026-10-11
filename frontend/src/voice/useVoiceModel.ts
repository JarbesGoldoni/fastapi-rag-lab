import { useCallback, useEffect, useRef, useState } from "react";
import { percent } from "../lib/format";
import type { FromWorker, ToWorker } from "./protocol";

export type ModelStatus = "idle" | "loading" | "ready" | "error";

type Pending = { resolve: (m: FromWorker) => void; reject: (e: Error) => void };
type Request = { type: "transcribe"; audio: Float32Array } | { type: "speak"; text: string; voice: string };

/**
 * Owns one voice worker: lazy loading, download progress, and request/response by id.
 * State lives in the hook (the imperative shell); the worker protocol is plain data.
 */
export function useVoiceModel(create: () => Worker) {
  const worker = useRef<Worker | null>(null);
  const pending = useRef(new Map<number, Pending>());
  const files = useRef(new Map<string, { loaded: number; total: number }>());
  const nextId = useRef(1);
  const loadSent = useRef(false);
  const [status, setStatus] = useState<ModelStatus>("idle");
  const [progress, setProgress] = useState(0);
  const [error, setError] = useState<string | null>(null);

  const ensure = useCallback((): Worker => {
    if (worker.current) return worker.current;
    const w = create();
    w.onmessage = (event: MessageEvent<FromWorker>) => {
      const message = event.data;
      switch (message.type) {
        case "progress": {
          files.current.set(message.file, { loaded: message.loaded, total: message.total });
          const all = [...files.current.values()];
          setProgress(
            percent(
              all.reduce((s, f) => s + f.loaded, 0),
              all.reduce((s, f) => s + f.total, 0),
            ),
          );
          break;
        }
        case "ready":
          setStatus("ready");
          setProgress(100);
          break;
        case "error":
          setError(message.message);
          if (message.id === undefined) setStatus("error");
          pending.current.get(message.id ?? -1)?.reject(new Error(message.message));
          pending.current.delete(message.id ?? -1);
          break;
        default:
          pending.current.get(message.id)?.resolve(message);
          pending.current.delete(message.id);
      }
    };
    worker.current = w;
    return w;
  }, [create]);

  // Guarded by refs, not state: React StrictMode remounts with stale state in development.
  const load = useCallback(() => {
    if (worker.current && loadSent.current) return;
    loadSent.current = true;
    setStatus("loading");
    setError(null);
    ensure().postMessage({ type: "load" } satisfies ToWorker);
  }, [ensure]);

  const request = useCallback(
    (body: Request): Promise<FromWorker> => {
      const id = nextId.current++;
      const w = ensure();
      setStatus((s) => (s === "idle" ? "loading" : s));
      return new Promise((resolve, reject) => {
        pending.current.set(id, { resolve, reject });
        const message = { ...body, id } as ToWorker;
        if (message.type === "transcribe") w.postMessage(message, [message.audio.buffer]);
        else w.postMessage(message);
      });
    },
    [ensure],
  );

  useEffect(
    () => () => {
      worker.current?.terminate();
      worker.current = null;
      loadSent.current = false;
      files.current.clear();
      pending.current.forEach((p) => p.reject(new Error("Voice model stopped")));
      pending.current.clear();
      setStatus("idle");
    },
    [],
  );

  return { status, progress, error, load, request };
}
