import { useCallback, useRef, useState } from "react";
import { decodeToMono16k, startRecording, type Recorder } from "../voice/audio";
import { useVoiceModel } from "../voice/useVoiceModel";

const createStt = () =>
  new Worker(new URL("../voice/stt.worker.ts", import.meta.url), { type: "module" });

export type Listening = "idle" | "recording" | "transcribing";

/** Mic button: the first press records, the second transcribes with Whisper in a worker. */
export function useDictation(onError: (message: string) => void, beforeRecording: () => void) {
  const stt = useVoiceModel(createStt);
  const { load, request } = stt;
  const recorder = useRef<Recorder | null>(null);
  const [listening, setListening] = useState<Listening>("idle");
  const [transcript, setTranscript] = useState<string | null>(null);

  const toggle = useCallback(async () => {
    const current = recorder.current;
    if (current) {
      recorder.current = null;
      setListening("transcribing");
      try {
        const audio = await decodeToMono16k(await current.stop());
        const reply = await request({ type: "transcribe", audio });
        if (reply.type === "text" && reply.text) setTranscript(reply.text);
      } catch (error) {
        onError(`Speech recognition failed: ${String(error)}`);
      } finally {
        setListening("idle");
      }
      return;
    }
    beforeRecording();
    load();
    try {
      recorder.current = await startRecording();
      setTranscript(null);
      setListening("recording");
    } catch {
      onError("Microphone access was denied.");
    }
  }, [load, request, onError, beforeRecording]);

  const progress = stt.status === "loading" ? stt.progress : null;
  return { listening, transcript, progress, toggle };
}
