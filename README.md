# RAG Lab

Ask questions about your PDFs, by typing or speaking, and get answers that cite their pages.
No API key needed: answers come from the Claude Code or Codex CLI already logged in on your
machine. Runs on macOS and Linux.

A practice project for an async Python stack: FastAPI, SQLAlchemy 2.0 async, Alembic,
PostgreSQL + pgvector, pytest. The UI is React + TypeScript.

## Run it

Needs [uv](https://docs.astral.sh/uv/), Node 20+, and the Claude Code or Codex CLI (logged in)
or an Anthropic/OpenAI API key.

```bash
docker compose up -d
```

```bash
cd backend && cp .env.example .env && uv sync && uv run alembic upgrade head && uv run raglab
```

```bash
cd frontend && npm install && npm run dev
```

Open http://localhost:5173 and click **Try the sample handbook**. To use your own Postgres
instead of Docker, install pgvector 0.8+ and set `DATABASE_URL` in `backend/.env`.

## How it works

**Ingestion** runs in a worker that reads jobs from a Postgres queue:

1. PyMuPDF extracts the text layer. AI reads the PDF only if it is scanned and you allow it.
2. The text is split into overlapping passages, each with an optional one-line context
   written by the model.
3. A local model (`bge-small-en-v1.5`) embeds the passages, and pgvector stores them.

The sidebar shows the real progress of each step (pages read, passages embedded).

**Answering**

1. A follow-up question ("why does it do that?") is first rewritten by the model into a
   standalone query.
2. Hybrid search runs vector similarity and Postgres full-text search, and fuses the two
   rankings (RRF).
3. The model answers from the numbered passages and cites them. The answer streams to the
   browser over server-sent events.

**Voice** runs in the browser: Whisper for speech-to-text, Kokoro for text-to-speech.

## Design

- **Functional core, imperative shell.** Chunking, ranking, prompts and parsing are pure
  functions. Pipelines receive their effects (PDF reader, embedder, model, database) as
  arguments, so tests use deterministic fakes and never call a model.
- **One composition root.** `wiring.Container` builds the adapters once and exposes the use
  cases; routes and the worker go through it.
- **A safe job queue.** Jobs are claimed with `FOR UPDATE SKIP LOCKED` and leased to one
  worker, which renews the lease on a heartbeat. If a worker dies, its job is retried, up
  to 3 attempts. Deleting or reprocessing a document stops its job within a heartbeat, and
  a stale job can never write again. Several workers can share the queue
  (`EMBEDDED_WORKER=false`, then `uv run raglab-worker`).
- **Async where it pays off:**
  - CPU work runs in `to_thread`.
  - The two searches run concurrently in a `TaskGroup`.
  - Batches use bounded concurrency (`Semaphore`).
  - CLI subprocesses are killed on timeout or when the browser disconnects.
- **Local but not open.** The CLIs run on your login, so the backend accepts writes only
  from its own origin and the UI's, and serves only `localhost` and `127.0.0.1`.

```
backend/src/raglab/
  core/        pure functions        pipelines/  ingest and answer, effects as arguments
  adapters/    PDF, embeddings, CLIs, APIs        db/  schema and queries
  wiring.py    composition root      worker.py   job queue consumer      api/  routes
frontend/src/
  lib/  pure helpers    hooks/  documents, chat, speech, dictation    voice/  model workers
```

## Tests

```bash
cd backend && TEST_DATABASE_URL=postgresql+asyncpg://raglab:raglab@localhost:5432/raglab_test uv run pytest -q
```

```bash
cd frontend && npm test
```

Without `TEST_DATABASE_URL`, the integration tests are skipped. Create and migrate that
database first.
`uv run python evals/run.py` measures retrieval recall on the sample handbook. CI runs ruff,
pyright (strict), the migrations and all tests on Ubuntu.

## Limits

- The Codex CLI adapter follows its documented JSON events but has not been run here.
- Gemini Nano appears only when Chrome offers its built-in model.
- Kokoro voices are English only.
- Changing the embedding model re-ingests every document at the next start.
