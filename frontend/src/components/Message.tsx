import { useState } from "react";
import { fileUrl } from "../lib/api";
import type { Message } from "../lib/chat";
import { segments } from "../lib/citations";
import type { Hit } from "../lib/types";

function Source({ number, hit }: { number: number; hit: Hit }) {
  return (
    <a className="source" href={fileUrl(hit.document_id, hit.page)} target="_blank" rel="noreferrer">
      <span className="source-number">{number}</span>
      <span className="source-body">
        <span className="source-title">
          {hit.filename} · page {hit.page}
        </span>
        <span className="source-text">{hit.text}</span>
      </span>
    </a>
  );
}

export function MessageView({ message, providerLabel }: { message: Message; providerLabel?: string }) {
  const [showSources, setShowSources] = useState(false);
  const hits = message.hits ?? [];

  if (message.role === "user") {
    return <div className="message user">{message.text}</div>;
  }

  const cited = new Set((message.citations ?? []).map((c) => c.number));
  const waiting = message.status === "streaming" && !message.text;

  return (
    <div className={`message assistant ${message.status}`}>
      <div className="answer">
        {waiting ? (
          <span className="thinking">
            <i />
            <i />
            <i />
          </span>
        ) : (
          segments(message.text).map((segment, i) =>
            segment.kind === "text" ? (
              <span key={i}>{segment.text}</span>
            ) : hits[segment.number - 1] ? (
              <a
                key={i}
                className="cite"
                href={fileUrl(hits[segment.number - 1]!.document_id, hits[segment.number - 1]!.page)}
                target="_blank"
                rel="noreferrer"
                title={`${hits[segment.number - 1]!.filename}, page ${hits[segment.number - 1]!.page}`}
              >
                {segment.number}
              </a>
            ) : null,
          )
        )}
        {message.status === "streaming" && message.text && <span className="caret" />}
      </div>
      {message.error && <p className="inline-error">{message.error}</p>}
      {hits.length > 0 && (
        <div className="message-footer">
          <button className="link" onClick={() => setShowSources((v) => !v)}>
            {showSources ? "Hide" : "Show"} {hits.length} sources
            {cited.size > 0 ? ` · ${cited.size} cited` : ""}
          </button>
          {providerLabel && <span className="muted">{providerLabel}</span>}
        </div>
      )}
      {showSources && (
        <div className="sources">
          {hits.map((hit, i) => (
            <Source key={hit.chunk_id} number={i + 1} hit={hit} />
          ))}
        </div>
      )}
    </div>
  );
}
