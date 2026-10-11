import { useEffect, useMemo, useRef, useState } from "react";
import { Composer } from "./components/Composer";
import { SpeakerIcon } from "./components/Icons";
import { MessageView } from "./components/Message";
import { Sidebar } from "./components/Sidebar";
import { useChat } from "./hooks/useChat";
import { useDictation } from "./hooks/useDictation";
import { useDocuments } from "./hooks/useDocuments";
import { useSpeech } from "./hooks/useSpeech";
import * as api from "./lib/api";
import { nanoAvailability } from "./lib/nano";
import type { Provider, Providers } from "./lib/types";

const NANO: Provider = { id: "chrome-nano", label: "Gemini Nano (in Chrome)", kind: "browser" };
const SUGGESTIONS = [
  "When is the on-call week handed over?",
  "How long can a card payment be refunded?",
  "What happens if I exceed the rate limit?",
];

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
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const [notice, setNotice] = useState<string | null>(null);
  const [backendError, setBackendError] = useState<string | null>(null);
  const [speak, setSpeak] = useState(recall("speak") === "1");
  const scroller = useRef<HTMLDivElement>(null);

  const speech = useSpeech(speak);
  const chat = useChat(speech);
  const { documents, upload, uploadSample, remove } = useDocuments(setNotice);
  const dictation = useDictation(setNotice, speech.stop);
  const { messages, busy } = chat;

  const generators = useMemo(
    () => [...(providers?.generators ?? []), ...(nano ? [NANO] : [])],
    [providers, nano],
  );
  const active = generators.find((g) => g.id === provider) ?? generators[0];
  const readyDocs = documents.filter((d) => d.status === "ready");

  useEffect(() => {
    api
      .getProviders()
      .then(setProviders)
      .catch(() => setBackendError("Can't reach the backend on port 8000. Is it running?"));
    nanoAvailability().then((a) => setNano(a !== "unavailable"));
  }, []);

  useEffect(() => {
    scroller.current?.scrollTo({ top: scroller.current.scrollHeight, behavior: "smooth" });
  }, [messages]);

  const ask = (question: string) => {
    if (active) void chat.ask(question, active, selected.size ? [...selected] : null);
  };

  const onDelete = async (id: string) => {
    await remove(id);
    setSelected((s) => new Set([...s].filter((x) => x !== id)));
  };

  const toggle = (id: string) =>
    setSelected((s) => {
      const next = new Set(s);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });

  // ------------------------------------------------------------ render

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
        onDelete={(id) => void onDelete(id)}
        onSample={uploadSample}
        uploadError={notice}
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
              <button className="ghost" onClick={chat.clear}>
                New chat
              </button>
            )}
            <button
              className={`toggle ${speak ? "on" : ""}`}
              onClick={() => {
                const next = !speak;
                setSpeak(next);
                remember("speak", next ? "1" : "0");
                if (!next) speech.stop();
              }}
              title="Read answers aloud"
            >
              <SpeakerIcon width={16} height={16} />
              <span>
                {speak && speech.status === "loading"
                  ? `Loading voice ${speech.progress}%`
                  : "Read aloud"}
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
                    <button key={s} onClick={() => ask(s)} disabled={busy || !active}>
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
          listening={dictation.listening}
          micProgress={dictation.progress}
          transcript={dictation.transcript}
          onSend={ask}
          onStop={chat.stop}
          onMic={() => void dictation.toggle()}
        />
      </main>
    </div>
  );
}
