import { useCallback, useEffect, useMemo, useReducer, useRef, useState } from "react";
import { Composer } from "./components/Composer";
import { SpeakerIcon } from "./components/Icons";
import { MessageView } from "./components/Message";
import { Sidebar } from "./components/Sidebar";
import * as api from "./lib/api";
import { chatReducer, historyOf } from "./lib/chat";
import { citationsOf } from "./lib/citations";
import { nanoAvailability, nanoStream } from "./lib/nano";
import { takeSentences } from "./lib/sentences";
import type { AskEvent, DocumentInfo, Provider, Providers } from "./lib/types";
import { createPlayer, decodeToMono16k, startRecording, type Recorder } from "./voice/audio";
import { useVoiceModel } from "./voice/useVoiceModel";

const NANO: Provider = { id: "chrome-nano", label: "Gemini Nano (in Chrome)", kind: "browser" };
const VOICE = "af_heart";
const SUGGESTIONS = [
  "When is the on-call week handed over?",
  "How long can a card payment be refunded?",
  "What happens if I exceed the rate limit?",
];

const createStt = () =>
  new Worker(new URL("./voice/stt.worker.ts", import.meta.url), { type: "module" });
const createTts = () =>
  new Worker(new URL("./voice/tts.worker.ts", import.meta.url), { type: "module" });

const remember = (key: string, value: string) => {
  try {
    localStorage.setItem(key, value);
  } catch {
    /* storage unavailable: ignore */
  }
};
const recall = (key: string) => {
  try {
    return localStorage.getItem(key);
  } catch {
    return null;
  }
};

export default function App() {
  const [providers, setProviders] = useState<Providers | null>(null);
  const [nano, setNano] = useState(false);
  const [provider, setProvider] = useState<string>(recall("provider") ?? "");
  const [documents, setDocuments] = useState<DocumentInfo[]>([]);
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const [messages, dispatch] = useReducer(chatReducer, []);
  const [uploadError, setUploadError] = useState<string | null>(null);
  const [backendError, setBackendError] = useState<string | null>(null);
  const [speak, setSpeak] = useState(recall("speak") === "1");
  const [listening, setListening] = useState<"idle" | "recording" | "transcribing">("idle");
  const [transcript, setTranscript] = useState<string | null>(null);

  const abort = useRef<AbortController | null>(null);
  const recorder = useRef<Recorder | null>(null);
  const player = useRef(createPlayer());
  const speech = useRef({ buffer: "", chain: Promise.resolve(), generation: 0 });
  const scroller = useRef<HTMLDivElement>(null);

  const stt = useVoiceModel(createStt);
  const tts = useVoiceModel(createTts);

  const generators = useMemo(
    () => [...(providers?.generators ?? []), ...(nano ? [NANO] : [])],
    [providers, nano],
  );
  const active = generators.find((g) => g.id === provider) ?? generators[0];
  const busy = messages.some((m) => m.status === "streaming");
  const readyDocs = documents.filter((d) => d.status === "ready");

  // ------------------------------------------------------------ loading

  useEffect(() => {
    api
      .getProviders()
      .then(setProviders)
      .catch(() => setBackendError("Can't reach the backend on port 8000. Is it running?"));
    nanoAvailability().then((a) => setNano(a !== "unavailable"));
  }, []);

  const refresh = useCallback(() => api.listDocuments().then(setDocuments).catch(() => {}), []);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  useEffect(() => {
    // Fast while something is being ingested, slow otherwise (uploads from other tabs or the API).
    const working = documents.some((d) => d.status === "queued" || d.status === "processing");
    const timer = setInterval(refresh, working ? 700 : 5000);
    return () => clearInterval(timer);
  }, [documents, refresh]);

  useEffect(() => {
    scroller.current?.scrollTo({ top: scroller.current.scrollHeight, behavior: "smooth" });
  }, [messages]);

  useEffect(() => {
    if (speak) tts.load();
  }, [speak, tts]);

  // ------------------------------------------------------------ speech out

  const stopSpeaking = useCallback(() => {
    speech.current.generation += 1;
    speech.current.buffer = "";
    speech.current.chain = Promise.resolve();
    player.current.stop();
  }, []);

  const say = useCallback(
    (sentences: string[]) => {
      const generation = speech.current.generation;
      for (const text of sentences) {
        // Synthesis runs one sentence at a time; playback is scheduled back to back.
        speech.current.chain = speech.current.chain
          .then(async () => {
            if (generation !== speech.current.generation) return;
            const reply = await tts.request({ type: "speak", text, voice: VOICE });
            if (reply.type === "audio" && generation === speech.current.generation)
              void player.current.play(reply.samples, reply.sampleRate);
          })
          .catch(() => {});
      }
    },
    [tts],
  );

  const feedSpeech = useCallback(
    (text: string, flush = false) => {
      if (!speak) return;
      const taken = takeSentences(speech.current.buffer + text, { flush });
      speech.current.buffer = taken.rest;
      say(taken.sentences);
    },
    [speak, say],
  );

  // ------------------------------------------------------------ asking

  const handleEvent = useCallback(
    (id: string, event: AskEvent) => {
      switch (event.type) {
        case "sources":
          return dispatch({ type: "sources", id, hits: event.hits });
        case "delta":
          feedSpeech(event.text);
          return dispatch({ type: "delta", id, text: event.text });
        case "reset":
          stopSpeaking();
          return dispatch({ type: "reset", id });
        case "done":
          feedSpeech("", true);
          return dispatch({ type: "done", id, answer: event.answer, citations: event.citations });
        case "error":
          return dispatch({ type: "error", id, message: event.message });
      }
    },
    [feedSpeech, stopSpeaking],
  );

  const ask = useCallback(
    async (question: string) => {
      if (!active) return;
      stopSpeaking();
      const id = crypto.randomUUID();
      const controller = new AbortController();
      abort.current = controller;
      const body = {
        query: question,
        history: historyOf(messages),
        document_ids: selected.size ? [...selected] : null,
      };
      dispatch({ type: "ask", id, question, provider: active.id });
      try {
        if (active.kind === "browser") {
          const context = await api.getContext(body);
          handleEvent(id, { type: "sources", hits: context.hits });
          let answer = "";
          for await (const delta of nanoStream(context.system, context.user, controller.signal, () => {})) {
            answer += delta;
            handleEvent(id, { type: "delta", text: delta });
          }
          handleEvent(id, { type: "done", answer, citations: citationsOf(answer, context.hits) });
        } else {
          await api.askStream({ ...body, provider: active.id }, (e) => handleEvent(id, e), controller.signal);
        }
      } catch (error) {
        if (controller.signal.aborted) dispatch({ type: "stop", id });
        else dispatch({ type: "error", id, message: String(error instanceof Error ? error.message : error) });
      } finally {
        dispatch({ type: "stop", id });
      }
    },
    [active, messages, selected, handleEvent, stopSpeaking],
  );

  const stop = () => {
    abort.current?.abort();
    stopSpeaking();
  };

  // ------------------------------------------------------------ speech in

  const toggleMic = async () => {
    if (listening === "recording" && recorder.current) {
      const current = recorder.current;
      recorder.current = null;
      setListening("transcribing");
      try {
        const audio = await decodeToMono16k(await current.stop());
        const reply = await stt.request({ type: "transcribe", audio });
        if (reply.type === "text" && reply.text) setTranscript(reply.text);
      } catch (error) {
        setUploadError(`Speech recognition failed: ${String(error)}`);
      } finally {
        setListening("idle");
      }
      return;
    }
    stopSpeaking();
    stt.load();
    try {
      recorder.current = await startRecording();
      setTranscript(null);
      setListening("recording");
    } catch {
      setUploadError("Microphone access was denied.");
    }
  };

  // ------------------------------------------------------------ documents

  const upload = async (files: File[], options: api.UploadOptions) => {
    setUploadError(null);
    for (const file of files) {
      try {
        await api.uploadDocument(file, file.name, options);
      } catch (error) {
        setUploadError(`${file.name}: ${error instanceof Error ? error.message : String(error)}`);
      }
    }
    await refresh();
  };

  const uploadSample = async () => {
    const blob = await fetch("/samples/northwind_handbook.pdf").then((r) => r.blob());
    await upload([new File([blob], "northwind_handbook.pdf", { type: "application/pdf" })], {});
  };

  const remove = async (id: string) => {
    await api.deleteDocument(id);
    setSelected((s) => new Set([...s].filter((x) => x !== id)));
    await refresh();
  };

  const toggle = (id: string) =>
    setSelected((s) => {
      const next = new Set(s);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });

  // ------------------------------------------------------------ render

  const micProgress = stt.status === "loading" ? stt.progress : null;
  const placeholder = readyDocs.length
    ? "Ask about your documents…"
    : documents.length
      ? "Reading your documents…"
      : "Upload a PDF to start";

  return (
    <div className="app">
      <Sidebar
        documents={documents}
        selected={selected}
        generators={generators}
        transcribers={providers?.transcribers ?? []}
        embeddingModel={providers?.embedding_model ?? ""}
        onToggle={toggle}
        onUpload={upload}
        onDelete={remove}
        onSample={uploadSample}
        uploadError={uploadError}
      />

      <main className="main">
        <header className="topbar">
          <label className="provider">
            <span>Answer with</span>
            <select
              value={active?.id ?? ""}
              onChange={(e) => {
                setProvider(e.target.value);
                remember("provider", e.target.value);
              }}
              disabled={!generators.length}
            >
              {generators.length === 0 && <option>No model found</option>}
              {generators.map((g) => (
                <option key={g.id} value={g.id}>
                  {g.label}
                </option>
              ))}
            </select>
          </label>
          <div className="topbar-right">
            {messages.length > 0 && (
              <button className="ghost" onClick={() => dispatch({ type: "clear" })}>
                New chat
              </button>
            )}
            <button
              className={`toggle ${speak ? "on" : ""}`}
              onClick={() => {
                const next = !speak;
                setSpeak(next);
                remember("speak", next ? "1" : "0");
                if (!next) stopSpeaking();
              }}
              title="Read answers aloud"
            >
              <SpeakerIcon width={16} height={16} />
              <span>
                {speak && tts.status === "loading" ? `Loading voice ${tts.progress}%` : "Read aloud"}
              </span>
            </button>
          </div>
        </header>

        {backendError && <div className="banner">{backendError}</div>}
        {providers && providers.generators.length === 0 && !nano && (
          <div className="banner">
            No model found. Install and log in to the Claude Code or Codex CLI, or set
            ANTHROPIC_API_KEY / OPENAI_API_KEY for the backend, then restart it.
          </div>
        )}

        <div className="conversation" ref={scroller}>
          {messages.length === 0 ? (
            <div className="empty">
              <h1>Ask your documents</h1>
              <p>
                Upload PDFs, then ask by typing or speaking. Answers cite the pages they come from.
              </p>
              {readyDocs.some((d) => d.filename.startsWith("northwind")) && (
                <div className="suggestions">
                  {SUGGESTIONS.map((s) => (
                    <button key={s} onClick={() => void ask(s)} disabled={busy || !active}>
                      {s}
                    </button>
                  ))}
                </div>
              )}
            </div>
          ) : (
            messages.map((m) => (
              <MessageView
                key={m.id}
                message={m}
                providerLabel={generators.find((g) => g.id === m.provider)?.label}
              />
            ))
          )}
        </div>

        <Composer
          busy={busy}
          disabled={!readyDocs.length || !active}
          placeholder={placeholder}
          listening={listening}
          micProgress={micProgress}
          transcript={transcript}
          onSend={(q) => void ask(q)}
          onStop={stop}
          onMic={() => void toggleMic()}
        />
      </main>
    </div>
  );
}
