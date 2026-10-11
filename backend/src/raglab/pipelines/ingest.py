"""Ingestion: extract -> clean -> chunk -> (contextualize) -> embed -> save.

Every effect is a parameter, so the same function runs in the worker, in tests with fakes,
and in the eval script.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Sequence

from raglab.core.chunking import chunk_pages, with_contexts
from raglab.core.prompts import context_prompt, parse_contexts
from raglab.core.text import clean_pages, total_chars
from raglab.domain import Chunk, ChunkConfig, IngestResult, Page, Progress, RawDocument, Stage
from raglab.pipelines.concurrency import batches, complete, ignore_progress, map_batched
from raglab.ports import Embedder, Extractor, Generator, IngestOptions, Report, SaveChunks

log = logging.getLogger(__name__)

DEFAULT_CHUNKING = ChunkConfig()


class IngestError(Exception):
    pass


async def contextualize(
    chunks: Sequence[Chunk],
    pages: Sequence[Page],
    generator: Generator,
    *,
    report: Report = ignore_progress,
    batch: int = 25,
    concurrency: int = 2,
    excerpt_chars: int = 12_000,
) -> tuple[tuple[Chunk, ...], tuple[str, ...]]:
    """Ask the model for one situating sentence per chunk, a batch of chunks per call.

    A failed batch keeps its chunks without context instead of failing the whole document.
    """
    excerpt = "\n\n".join(p.text for p in pages)[:excerpt_chars]
    semaphore = asyncio.Semaphore(concurrency)
    done = 0

    async def ask(part: Sequence[Chunk]) -> tuple[str, ...]:
        async with semaphore:
            try:
                text = await complete(generator.stream, context_prompt(excerpt, part))
            except Exception as exc:
                log.warning("contextualize batch failed: %s", exc)
                return ("",) * len(part)
        contexts = parse_contexts(text, expected=len(part))
        return contexts + ("",) * (len(part) - len(contexts))

    async def run(part: Sequence[Chunk]) -> tuple[str, ...]:
        nonlocal done
        contexts = await ask(part)
        done += len(part)
        await report(Progress(Stage.CONTEXT, done, len(chunks)))
        return contexts

    await report(Progress(Stage.CONTEXT, 0, len(chunks)))

    async with asyncio.TaskGroup() as group:
        tasks = [group.create_task(run(part)) for part in batches(chunks, batch)]
    contexts = tuple(c for task in tasks for c in task.result())
    missing = sum(1 for c in contexts if not c)
    warnings = (f"{missing} of {len(chunks)} chunks have no context",) if missing else ()
    return with_contexts(chunks, contexts), warnings


async def ingest(
    doc: RawDocument,
    options: IngestOptions,
    *,
    extractor: Extractor,
    embedder: Embedder,
    save: SaveChunks,
    cfg: ChunkConfig = DEFAULT_CHUNKING,
    min_chars_per_page: int = 20,
    embed_batch: int = 64,
    embed_concurrency: int = 2,
    report: Report = ignore_progress,
) -> IngestResult:
    raw_pages = await extractor.extract(doc, report)
    pages = clean_pages(raw_pages)
    used = extractor.id
    warnings: tuple[str, ...] = ()

    # A scanned PDF has pages but (almost) no text layer.
    if total_chars(pages) < min_chars_per_page * max(1, len(raw_pages)):
        if options.ai_extractor is None:
            warnings += ("The PDF has little or no text layer (scanned?). Enable AI extraction.",)
        else:
            pages = clean_pages(await options.ai_extractor.extract(doc, report))
            used = options.ai_extractor.id

    if not pages:
        raise IngestError("No text found in the document.")

    chunks = chunk_pages(doc.id, pages, cfg=cfg)
    if options.contextualize_with is not None:
        chunks, context_warnings = await contextualize(
            chunks, pages, options.contextualize_with, report=report
        )
        warnings += context_warnings

    async def embedded(done: int, total: int) -> None:
        await report(Progress(Stage.EMBEDDING, done, total))

    await embedded(0, len(chunks))
    vectors = await map_batched(
        embedder.documents,
        [c.embedding_text for c in chunks],
        batch=embed_batch,
        concurrency=embed_concurrency,
        on_progress=embedded,
    )
    await report(Progress(Stage.SAVING, 0, len(chunks)))
    await save(doc.id, chunks, vectors, embedder.model)
    await report(Progress(Stage.SAVING, len(chunks), len(chunks)))
    return IngestResult(
        document_id=doc.id,
        pages=len(pages),
        chunks=len(chunks),
        extractor=used,
        contextualized=options.contextualize_with is not None,
        warnings=warnings,
    )
