/// <reference lib="webworker" />
/**
 * Speech to text with Whisper, running in the browser via transformers.js (WASM, any OS).
 * The model (~80 MB) downloads once and is cached by the browser.
 */
import { pipeline } from "@huggingface/transformers";
import { progressMessage, type FromWorker, type ToWorker } from "./protocol";

const MODEL = "onnx-community/whisper-base";

type Transcriber = (audio: Float32Array, options: object) => Promise<{ text: string } | { text: string }[]>;

let model: Promise<Transcriber> | null = null;

const send = (message: FromWorker) => postMessage(message);

function load(): Promise<Transcriber> {
  model ??= pipeline("automatic-speech-recognition", MODEL, {
    dtype: "q8",
    device: "wasm",
    progress_callback: (info) => {
      const message = progressMessage(info as { status: string });
      if (message) send(message);
    },
  }).then((p) => {
    send({ type: "ready" });
    return p as unknown as Transcriber;
  });
  return model;
}

self.onmessage = async (event: MessageEvent<ToWorker>) => {
  const message = event.data;
  try {
    if (message.type === "load") await load();
    if (message.type === "transcribe") {
      const transcribe = await load();
      const output = await transcribe(message.audio, { task: "transcribe", chunk_length_s: 30 });
      const text = (Array.isArray(output) ? output.map((o) => o.text).join(" ") : output.text).trim();
      send({ type: "text", id: message.id, text });
    }
  } catch (error) {
    model = null;
    send({ type: "error", id: "id" in message ? message.id : undefined, message: String(error) });
  }
};
