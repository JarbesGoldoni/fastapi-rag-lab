import { useCallback, useReducer, useRef } from "react";
import * as api from "../lib/api";
import { chatReducer, historyOf } from "../lib/chat";
import { citationsOf } from "../lib/citations";
import { nanoStream } from "../lib/nano";
import type { AskEvent, Provider } from "../lib/types";

interface Speech {
  feed: (text: string, flush?: boolean) => void;
  stop: () => void;
}

const collect = async (stream: AsyncGenerator<string>) => {
  let text = "";
  for await (const delta of stream) text += delta;
  return text;
};

/** The conversation: asks a backend model over SSE, or Gemini Nano in the browser. */
export function useChat(speech: Speech) {
  const [messages, dispatch] = useReducer(chatReducer, []);
  const abort = useRef<AbortController | null>(null);
  const { feed, stop: stopSpeaking } = speech;

  const handle = useCallback(
    (id: string, event: AskEvent) => {
      switch (event.type) {
        case "sources":
          return dispatch({ type: "sources", id, hits: event.hits });
        case "delta":
          feed(event.text);
          return dispatch({ type: "delta", id, text: event.text });
        case "reset":
          stopSpeaking();
          return dispatch({ type: "reset", id });
        case "done":
          feed("", true);
          return dispatch({ type: "done", id, answer: event.answer, citations: event.citations });
        case "error":
          return dispatch({ type: "error", id, message: event.message });
      }
    },
    [feed, stopSpeaking],
  );

  const askNano = async (id: string, body: api.AskBody, signal: AbortSignal) => {
    // A follow-up is rewritten into a standalone query first, as the backend does.
    let rewritten: string | undefined;
    if (body.history.length) {
      const prompt = await api.getRewritePrompt(body);
      rewritten = await collect(nanoStream(prompt.system, prompt.user, signal, () => {}));
    }
    const context = await api.getContext({ ...body, rewritten });
    handle(id, { type: "sources", hits: context.hits });
    let answer = "";
    for await (const delta of nanoStream(context.system, context.user, signal, () => {})) {
      answer += delta;
      handle(id, { type: "delta", text: delta });
    }
    handle(id, { type: "done", answer, citations: citationsOf(answer, context.hits) });
  };

  const ask = async (question: string, provider: Provider, documentIds: string[] | null) => {
    stopSpeaking();
    const id = crypto.randomUUID();
    const controller = new AbortController();
    abort.current = controller;
    const body = { query: question, history: historyOf(messages), document_ids: documentIds };
    dispatch({ type: "ask", id, question, provider: provider.id });
    try {
      if (provider.kind === "browser") await askNano(id, body, controller.signal);
      else
        await api.askStream({ ...body, provider: provider.id }, (e) => handle(id, e), controller.signal);
    } catch (error) {
      if (!controller.signal.aborted)
        dispatch({ type: "error", id, message: error instanceof Error ? error.message : String(error) });
    } finally {
      dispatch({ type: "stop", id });
    }
  };

  const stop = () => {
    abort.current?.abort();
    stopSpeaking();
  };

  return {
    messages,
    busy: messages.some((m) => m.status === "streaming"),
    ask,
    stop,
    clear: () => dispatch({ type: "clear" }),
  };
}
