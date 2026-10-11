"""Ingestion worker: pulls jobs from the Postgres queue and runs the ingest pipeline.

Runs inside the API process by default (EMBEDDED_WORKER=true) or on its own:
    uv run raglab-worker

Each running job is leased to one worker, which renews the lease on a heartbeat. A job whose
worker stops renewing goes back to the queue; a worker that loses its job (deleted, requeued,
or taken over) cancels the work within one heartbeat.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import signal
import socket
import time
import uuid
from collections.abc import Awaitable, Callable

from raglab.db.repository import (
    Job,
    JobCancelled,
    claim_job,
    finish_job,
    get_raw_document,
    recover_expired_jobs,
    renew_lease,
    requeue_stale_embeddings,
    set_progress,
)
from raglab.domain import Progress
from raglab.pipelines.concurrency import guarded, throttled
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


def new_worker_id() -> str:
    return f"{socket.gethostname()}:{os.getpid()}:{uuid.uuid4().hex[:8]}"


async def keep_lease(container: Container, job: Job) -> None:
    """Renew the job's lease on every heartbeat; return once the job is no longer ours.

    A failed renewal (database briefly unreachable) is retried until the lease would have
    expired; after that another worker may own the job, so we stop."""
    settings = container.settings
    lease, heartbeat = settings.worker_lease_seconds, settings.worker_heartbeat_seconds
    expires = time.monotonic() + lease
    while True:
        try:
            if not await renew_lease(container.engine, job, lease_seconds=lease):
                return
            expires = time.monotonic() + lease
        except Exception as exc:
            if time.monotonic() >= expires:
                log.warning("lost the lease of job %d: %s", job.id, exc)
                return
            log.warning("could not renew the lease of job %d, retrying: %s", job.id, exc)
        await asyncio.sleep(heartbeat)


async def process(container: Container, job: Job) -> None:
    doc = await get_raw_document(container.engine, job.document_id)
    if doc is None:  # deleted while queued
        await finish_job(container.engine, job, error="document deleted", document_fields={})
        return
    options = container.ingest_options(job.options)

    async def store(progress: Progress) -> None:
        await set_progress(container.engine, job, progress)

    try:
        result = await guarded(
            container.ingest(doc, options, throttled(store, interval=0.3), job=job),
            keep_lease(container, job),
            lost=lambda: JobCancelled(job.id),
        )
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
    worker: str | None = None,
) -> None:
    """Claim and process jobs until `stop` is set. No single failure ends the loop: a job
    that could not be finished keeps its lease until it expires, then it is retried."""
    engine, settings = container.engine, container.settings
    worker = worker or new_worker_id()
    try:
        if stale := await requeue_stale_embeddings(engine, container.embedder.model):
            log.info("re-ingesting %d documents embedded with another model", stale)
    except Exception:
        log.exception("could not check for documents embedded with another model")
    while not stop.is_set():
        try:
            if recovered := await recover_expired_jobs(
                engine, max_attempts=settings.worker_max_attempts
            ):
                log.info("recovered %d jobs whose worker stopped", recovered)
            job = await claim_job(engine, worker, lease_seconds=settings.worker_lease_seconds)
        except Exception:
            log.exception("could not claim a job")
            job = None
        if job is None:
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(stop.wait(), settings.worker_poll_seconds)
            continue
        try:
            await handle(container, job)
        except Exception:
            log.exception("job %d failed outside ingestion; it will be retried", job.id)


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
