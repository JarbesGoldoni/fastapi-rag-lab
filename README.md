# RAG Lab

Upload PDFs, ask questions by typing or speaking, and get answers that cite the pages they
come from, read aloud if you want. Runs on macOS and Linux. No API key is required: answers
come from the Claude Code or Codex CLI already logged in on your machine. API keys
(Anthropic, OpenAI) are optional.

This is a practice project for a Python backend stack: FastAPI (async), SQLAlchemy 2.0 async,
Alembic, PostgreSQL with pgvector, pytest. The UI is React + TypeScript.

## How it works

```
Browser (React)                                   Python backend (FastAPI, runs on the host)
mic → Whisper (transformers.js, WASM) → question
upload PDF ─────────────────────────────────────► POST /api/documents → job queue (Postgres)
                                                    worker: PyMuPDF text → chunks
                                                    → (optional) AI context per chunk
                                                    → local embeddings (fastembed) → pgvector
question ───────────────────────────────────────► POST /api/ask
                                                    hybrid search: pgvector + full-text, fused (RRF)
                                                    → prompt with numbered sources
                                                    → claude -p / codex exec / API, streamed
◄──────────── server-sent events: sources, text, citations ──
answer → Kokoro TTS (kokoro-js, WASM), sentence by sentence
```

- **Reading PDFs uses no AI.** PyMuPDF extracts the text layer. For scanned PDFs, you can
  turn on "Read scanned PDFs with AI" (Claude CLI or Anthropic API).
- **Embeddings are local** (`BAAI/bge-small-en-v1.5`, ~70 MB, CPU). An LLM CLI cannot
  produce embeddings, so this part always uses an embedding model (local, or OpenAI with a key).
- **Search is hybrid**: vector similarity for meaning and Postgres full-text for exact words
  (names, codes, numbers), combined by reciprocal rank fusion.
- **Answer models**, chosen per question in the UI:
  - Claude Code CLI (`claude -p`, tools off, streamed)
  - Codex CLI (`codex exec`, read-only sandbox)
  - Anthropic API or OpenAI API, when a key is set
  - Gemini Nano inside Chrome, when the browser offers it
- **Voice runs in the browser** on any OS: Whisper for speech-to-text and Kokoro for
  text-to-speech. The models download on first use (~80 MB and ~90 MB) and are cached.

## Run it

Requirements: Python tooling [uv](https://docs.astral.sh/uv/), Node 20+, PostgreSQL 15+ with
pgvector, and at least one of: Claude Code CLI or Codex CLI (logged in), or an API key.

```bash
docker compose up -d
```

(or use a local Postgres with `pgvector` installed, and point `DATABASE_URL` at it)

```bash
cd backend && cp .env.example .env && uv sync && uv run alembic upgrade head
```

```bash
cd backend && uv run raglab
```

```bash
cd frontend && npm install && npm run dev
```

Open http://localhost:5173 and click **Try the sample handbook**, or drop your own PDFs.

While a document is ingested, the sidebar shows the current stage with its measured count
and percentage: pages read, passages contextualized, passages embedded, saving. Each number
comes from the worker (one update per batch, at most about three per second); only an AI
transcription of a scanned PDF shows an indeterminate bar, because the page count is unknown
until the model replies. Deleting or reprocessing a document stops its running job at the
next update, and a stale job can never write to the document again.

The ingestion worker runs inside the API process by default. To run it separately, set
`EMBEDDED_WORKER=false` and start `uv run raglab-worker`; several workers can share the queue.

## Code layout

```
backend/src/raglab/
├── domain.py        frozen dataclasses: Page, Chunk, Hit, Prompt, Citation…
├── core/            pure functions: cleaning, chunking, RRF, prompts, citation parsing, SSE
├── ports.py         the effects pipelines need, as function types (Embedder, Generator…)
├── pipelines/       ingest and ask: compose core functions, receive effects as arguments
├── adapters/        PyMuPDF, fastembed, OpenAI embeddings, CLIs, Anthropic/OpenAI APIs
├── db/              schema and repository (SQLAlchemy Core, async)
├── wiring.py        composition root: picks adapters once at startup
├── worker.py        job queue consumer (SELECT … FOR UPDATE SKIP LOCKED)
└── api/             FastAPI routes and schemas
frontend/src/
├── lib/             pure helpers (SSE parser, sentence splitter, chat reducer) + API client
├── voice/           Whisper and Kokoro workers, recording and playback
└── components/      sidebar, messages, composer
```

**Functional core, imperative shell.** Chunking, ranking, prompt building and parsing are
pure and deterministic. Effects (PDF parsing, embeddings, model calls, database) are passed
into the pipelines as functions, so tests swap them for deterministic fakes and no test
calls a model. No module holds mutable global state; settings are read once and passed down.

### Where async is used

- FastAPI endpoints and SQLAlchemy async with asyncpg.
- CPU-bound work (PDF parsing, local embeddings) runs in `asyncio.to_thread`.
- Vector search and full-text search run concurrently in a `TaskGroup`.
- Embedding batches and contextualization calls run with bounded concurrency (`Semaphore`).
- The CLIs run with `asyncio.create_subprocess_exec` under `asyncio.timeout`. The process
  is killed when the browser disconnects, and stderr is drained concurrently.
- Answers stream token by token through an async generator into server-sent events.
  If the model restarts its answer (a CLI retry), a `reset` event clears the partial text.

## Tests and evaluation

```bash
cd backend && uv run pytest -q
```

Integration tests need a migrated database: set
`TEST_DATABASE_URL=postgresql+asyncpg://…/raglab_test`. They are skipped without it.

```bash
cd backend && uv run python evals/run.py
```

The eval ingests the sample handbook and measures recall@k for vector-only, keyword-only and
hybrid search over 22 questions. The sample is small (one passage per page), so scores near
100% are expected; the script is there to compare changes on harder documents.

```bash
cd frontend && npm test
```

CI (GitHub Actions on Ubuntu) runs ruff, pyright (strict), the migrations, all backend tests
against pgvector, and the frontend tests and build.

## Notes and limits

- The Codex CLI adapter follows the `codex exec --json` event format but was not tested here.
- Gemini Nano depends on the Chrome version and the machine; the option only appears when
  `LanguageModel` is available.
- Kokoro voices are English. Whisper detects the spoken language on its own.
- Switching the embedding model requires re-ingesting documents (chunks record their model).
- Uploaded PDFs are stored in Postgres; this is a lab setup, not a document store.
