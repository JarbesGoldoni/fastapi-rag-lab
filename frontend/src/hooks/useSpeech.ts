import { useCallback, useEffect, useRef } from "react";
import { takeSentences } from "../lib/sentences";
import { createPlayer } from "../voice/audio";
import { useVoiceModel } from "../voice/useVoiceModel";

const VOICE = "af_heart";
const createTts = () =>
  new Worker(new URL("../voice/tts.worker.ts", import.meta.url), { type: "module" });

/** Reads streamed text aloud sentence by sentence: synthesis one at a time, playback gapless. */
export function useSpeech(enabled: boolean) {
  const tts = useVoiceModel(createTts);
  const { load, request } = tts;
  const player = useRef(createPlayer());
  const speech = useRef({ buffer: "", chain: Promise.resolve(), generation: 0 });

  useEffect(() => {
    if (enabled) load();
  }, [enabled, load]);

  const stop = useCallback(() => {
    speech.current.generation += 1;
    speech.current.buffer = "";
    speech.current.chain = Promise.resolve();
    player.current.stop();
  }, []);

  const say = useCallback(
    (sentences: string[]) => {
      const generation = speech.current.generation;
      for (const text of sentences) {
        speech.current.chain = speech.current.chain
          .then(async () => {
            if (generation !== speech.current.generation) return;
            const reply = await request({ type: "speak", text, voice: VOICE });
            if (reply.type === "audio" && generation === speech.current.generation)
              void player.current.play(reply.samples, reply.sampleRate);
          })
          .catch(() => {});
      }
    },
    [request],
  );

  /** Feed streamed text; `flush` speaks the unfinished last sentence too. */
  const feed = useCallback(
    (text: string, flush = false) => {
      if (!enabled) return;
      const taken = takeSentences(speech.current.buffer + text, { flush });
      speech.current.buffer = taken.rest;
      say(taken.sentences);
    },
    [enabled, say],
  );

  return { feed, stop, status: tts.status, progress: tts.progress };
}
