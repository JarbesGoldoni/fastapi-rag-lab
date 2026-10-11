import { useRef, useState, type DragEvent } from "react";
import { fileUrl } from "../lib/api";
import { fileSize } from "../lib/format";
import { progressView } from "../lib/progress";
import type { DocumentInfo, Provider } from "../lib/types";
import { ExternalIcon, FileIcon, SparkIcon, TrashIcon, UploadIcon } from "./Icons";

interface Props {
  documents: DocumentInfo[];
  selected: Set<string>;
  generators: Provider[];
  transcribers: string[];
  embeddingModel: string;
  onToggle: (id: string) => void;
  onUpload: (files: File[], options: { contextualizeWith: string | null; aiExtraction: string | null }) => void;
  onDelete: (id: string) => void;
  onSample: () => void;
  uploadError: string | null;
}

const STATUS_LABEL: Record<DocumentInfo["status"], string> = {
  queued: "Queued",
  processing: "Processing",
  ready: "Ready",
  failed: "Failed",
};

export function Sidebar(props: Props) {
  const input = useRef<HTMLInputElement>(null);
  const [dragging, setDragging] = useState(false);
  const serverModels = props.generators.filter((g) => g.kind !== "browser");
  const [contextWith, setContextWith] = useState<string>("");
  const [extractWith, setExtractWith] = useState<string>("");

  const send = (files: File[]) => {
    const pdfs = files.filter((f) => f.type === "application/pdf" || f.name.endsWith(".pdf"));
    if (pdfs.length)
      props.onUpload(pdfs, { contextualizeWith: contextWith || null, aiExtraction: extractWith || null });
  };

  const onDrop = (event: DragEvent) => {
    event.preventDefault();
    setDragging(false);
    send([...event.dataTransfer.files]);
  };

  return (
    <aside className="sidebar">
      <div className="brand">
        <img src="/favicon.svg" alt="" width={26} height={26} />
        <div>
          <strong>RAG Lab</strong>
          <span>Ask your PDFs</span>
        </div>
      </div>

      <div
        className={`dropzone${dragging ? " dragging" : ""}`}
        onDragOver={(e) => {
          e.preventDefault();
          setDragging(true);
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={onDrop}
        onClick={() => input.current?.click()}
        role="button"
        tabIndex={0}
        onKeyDown={(e) => e.key === "Enter" && input.current?.click()}
      >
        <UploadIcon width={22} height={22} />
        <p>
          <strong>Drop PDFs here</strong>
          <span>or click to choose</span>
        </p>
        <input
          ref={input}
          type="file"
          accept="application/pdf"
          multiple
          hidden
          onChange={(e) => {
            send([...(e.target.files ?? [])]);
            e.target.value = "";
          }}
        />
      </div>

      {serverModels.length > 0 && (
        <details className="options">
          <summary>
            <SparkIcon width={15} height={15} /> AI options for new uploads
          </summary>
          <label>
            <span>Add context to each passage</span>
            <select value={contextWith} onChange={(e) => setContextWith(e.target.value)}>
              <option value="">Off</option>
              {serverModels.map((g) => (
                <option key={g.id} value={g.id}>
                  {g.label}
                </option>
              ))}
            </select>
          </label>
          {props.transcribers.length > 0 && (
            <label>
              <span>Read scanned PDFs with AI</span>
              <select value={extractWith} onChange={(e) => setExtractWith(e.target.value)}>
                <option value="">Off</option>
                {props.transcribers.map((id) => (
                  <option key={id} value={id}>
                    {serverModels.find((g) => g.id === id)?.label ?? id}
                  </option>
                ))}
              </select>
            </label>
          )}
        </details>
      )}

      {props.uploadError && <p className="inline-error">{props.uploadError}</p>}

      <div className="doc-header">
        <h2>Documents</h2>
        {props.selected.size > 0 && <span className="pill">{props.selected.size} selected</span>}
      </div>

      <ul className="documents">
        {props.documents.length === 0 && (
          <li className="doc-empty">
            No documents yet.{" "}
            <button className="link" onClick={props.onSample}>
              Try the sample handbook
            </button>
          </li>
        )}
        {props.documents.map((doc) => (
          <li key={doc.id} className={`doc ${props.selected.has(doc.id) ? "selected" : ""}`}>
            <label className="doc-main" title={doc.status === "ready" ? "Limit answers to this document" : ""}>
              <input
                type="checkbox"
                checked={props.selected.has(doc.id)}
                disabled={doc.status !== "ready"}
                onChange={() => props.onToggle(doc.id)}
              />
              <FileIcon className="doc-icon" />
              <span className="doc-text">
                <span className="doc-name">{doc.filename}</span>
                <span className="doc-meta">
                  <span className={`status status-${doc.status}`}>{STATUS_LABEL[doc.status]}</span>
                  {doc.pages !== null && <span>{doc.pages} pages</span>}
                  {doc.chunks !== null && <span>{doc.chunks} passages</span>}
                  <span>{fileSize(doc.size_bytes)}</span>
                </span>
                <ProgressRow doc={doc} />
                {doc.error && <span className="doc-warning">{doc.error}</span>}
                {doc.warnings.map((w) => (
                  <span key={w} className="doc-warning">
                    {w}
                  </span>
                ))}
              </span>
            </label>
            <div className="doc-actions">
              <a href={fileUrl(doc.id)} target="_blank" rel="noreferrer" title="Open PDF">
                <ExternalIcon width={15} height={15} />
              </a>
              <button title="Delete" onClick={() => props.onDelete(doc.id)}>
                <TrashIcon width={15} height={15} />
              </button>
            </div>
          </li>
        ))}
      </ul>

      <footer className="sidebar-footer">
        Search: hybrid (meaning + keywords)
        <br />
        Embeddings: {props.embeddingModel || "…"}
      </footer>
    </aside>
  );
}

function ProgressRow({ doc }: { doc: DocumentInfo }) {
  const view = progressView(doc);
  if (!view) return null;
  return (
    <span className="progress" role="status" aria-live="polite">
      <span className="progress-text">
        <span>{view.label}</span>
        {view.percent !== null && <strong>{view.percent}%</strong>}
      </span>
      <span
        className={`progress-track${view.percent === null ? " indeterminate" : ""}`}
        role="progressbar"
        aria-valuemin={0}
        aria-valuemax={100}
        aria-valuenow={view.percent ?? undefined}
      >
        <span className="progress-fill" style={{ width: `${view.percent ?? 30}%` }} />
      </span>
      {view.count && <span className="progress-count">{view.count}</span>}
    </span>
  );
}
