"""Ingestion worker: pulls jobs from the Postgres queue and runs the ingest pipeline.

Runs inside the API process by default (EMBEDDED_WORKER=true) or on its own:
    uv run python -m raglab.worker
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import signal
from collections.abc import Awaitable, Callable

from raglab.db.repository import (
    Job,
    JobCancelled,
    claim_job,
    finish_job,
    get_raw_document,
    recover_running_jobs,
    set_progress,
)
from raglab.domain import Progress
from raglab.pipelines.concurrency import throttled
from raglab.wiring import Container

log = logging.getLogger("raglab.worker")


def leaves(exc: BaseException) -> list[BaseException]:
    """The real exceptions inside (possibly nested) exception groups from TaskGroups."""
    if isinstance(exc, BaseExceptionGroup):
        return [leaf for inner in exc.exceptions for leaf in leaves(inner)]
    return [exc]


def is_cancelled(exc: BaseException) -> bool:
    return any(isinstance(leaf, JobCancelled) for leaf in leaves(exc))


def root_message(exc: BaseException) -> str:
    """A message a user can act on, not "unhandled errors in a TaskGroup"."""
    first = leaves(exc)[0]
    return str(first) or type(first).__name__


async def process(container: Container, job: Job) -> None:
    doc = await get_raw_document(container.engine, job.document_id)
    if doc is None:  # deleted while queued
        await finish_job(container.engine, job, error="document deleted", document_fields={})
        return
    options = container.ingest_options(job.options)

    async def store(progress: Progress) -> None:
        await set_progress(container.engine, job, progress)

    try:
        result = await container.ingest(doc, options, throttled(store, interval=0.3), job_id=job.id)
    except Exception as exc:
        if is_cancelled(exc):
            log.info("stopped %s: the document was deleted or requeued", doc.filename)
            return
        log.exception("ingest failed for %s", doc.filename)
        await finish_job(container.engine, job, error=root_message(exc), document_fields={})
        return
    await finish_job(
        container.engine,
        job,
        error=None,
        document_fields={
            "pages": result.pages,
            "chunks": result.chunks,
            "extractor": result.extractor,
            "contextualized": job.options.get("contextualize_with")
            if result.contextualized
            else None,
            "warnings": list(result.warnings),
        },
    )
    log.info("ingested %s: %d pages, %d chunks", doc.filename, result.pages, result.chunks)


async def run(
    container: Container,
    stop: asyncio.Event,
    *,
    handle: Callable[[Container, Job], Awaitable[None]] = process,
) -> None:
    recovered = await recover_running_jobs(container.engine)
    if recovered:
        log.info("requeued %d interrupted jobs", recovered)
    while not stop.is_set():
        try:
            job = await claim_job(container.engine)
        except Exception:
            log.exception("could not claim a job")
            job = None
        if job is None:
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(stop.wait(), container.settings.worker_poll_seconds)
            continue
        await handle(container, job)


async def main() -> None:
    from raglab.settings import Settings
    from raglab.wiring import build

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    container = await build(Settings())
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, stop.set)
    log.info("worker started")
    try:
        await run(container, stop)
    finally:
        await container.engine.dispose()


def cli() -> None:
    asyncio.run(main())


if __name__ == "__main__":
    cli()
