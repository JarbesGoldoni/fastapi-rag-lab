/// <reference lib="webworker" />
/**
 * Text to speech with Kokoro-82M, running in the browser via kokoro-js (WASM, any OS).
 * The model (~90 MB at q8) downloads once and is cached by the browser.
 */
import { KokoroTTS } from "kokoro-js";
import { progressMessage, type FromWorker, type ToWorker } from "./protocol";

const MODEL = "onnx-community/Kokoro-82M-v1.0-ONNX";

let model: Promise<KokoroTTS> | null = null;

const send = (message: FromWorker, transfer: Transferable[] = []) => postMessage(message, transfer);

function load(): Promise<KokoroTTS> {
  model ??= KokoroTTS.from_pretrained(MODEL, {
    dtype: "q8",
    device: "wasm",
    progress_callback: (info) => {
      const message = progressMessage(info as { status: string });
      if (message) send(message);
    },
  }).then((tts) => {
    send({ type: "ready" });
    return tts;
  });
  return model;
}

self.onmessage = async (event: MessageEvent<ToWorker>) => {
  const message = event.data;
  try {
    if (message.type === "load") await load();
    if (message.type === "speak") {
      const tts = await load();
      const audio = await tts.generate(message.text, { voice: message.voice as "af_heart" });
      const samples = audio.audio as Float32Array;
      send(
        { type: "audio", id: message.id, samples, sampleRate: audio.sampling_rate },
        [samples.buffer],
      );
    }
  } catch (error) {
    model = null;
    send({ type: "error", id: "id" in message ? message.id : undefined, message: String(error) });
  }
};
