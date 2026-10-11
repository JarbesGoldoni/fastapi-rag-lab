"""Postgres access. Plain async functions over an AsyncEngine, plus factories that bind the
engine and return the port-shaped functions the pipelines expect."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from sqlalchemy import delete, exists, func, insert, select, text, update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncConnection, AsyncEngine

from raglab.db.schema import chunks, documents, jobs
from raglab.domain import Chunk, DocumentStatus, Hit, Progress, RawDocument, Vector
from raglab.ports import SaveChunks, TextSearch, VectorSearch


@dataclass(frozen=True, slots=True)
class DocumentRow:
    id: str
    filename: str
    size_bytes: int
    status: str
    pages: int | None
    chunks: int | None
    extractor: str | None
    contextualized: str | None
    warnings: list[str]
    error: str | None
    progress_stage: str | None
    progress_done: int | None
    progress_total: int | None
    created_at: datetime


@dataclass(frozen=True, slots=True)
class Job:
    id: int
    document_id: str
    options: dict[str, Any]
    attempts: int
    worker: str


class JobCancelled(Exception):
    """The job is no longer this worker's: its document was deleted or requeued, or its
    lease expired and another worker took it. Stop working on it."""


def _owned(job: Job) -> Any:
    return (jobs.c.id == job.id) & (jobs.c.status == "running") & (jobs.c.worker_id == job.worker)


_DOCUMENT_COLUMNS = [c for c in documents.c if c.name not in {"content", "updated_at"}]


def vector_literal(vector: Vector) -> str:
    return "[" + ",".join(f"{x:.7g}" for x in vector) + "]"


def _document(row: Any) -> DocumentRow:
    return DocumentRow(**{c.name: getattr(row, c.name) for c in _DOCUMENT_COLUMNS})


# ---------------------------------------------------------------- documents


async def create_document(
    engine: AsyncEngine, doc: RawDocument, options: dict[str, Any]
) -> tuple[DocumentRow, bool]:
    """Insert the document and queue its ingestion in one transaction.

    Returns (row, created). An identical file uploaded again returns the existing row.
    """
    async with engine.begin() as conn:
        inserted = await conn.execute(
            pg_insert(documents)
            .values(
                id=doc.id,
                filename=doc.filename,
                content=doc.content,
                size_bytes=len(doc.content),
                status=DocumentStatus.QUEUED,
            )
            .on_conflict_do_nothing(index_elements=["id"])
            .returning(documents.c.id)
        )
        created = inserted.first() is not None
        if created:
            await conn.execute(insert(jobs).values(document_id=doc.id, options=options))
        row = (await conn.execute(select(*_DOCUMENT_COLUMNS).where(documents.c.id == doc.id))).one()
    return _document(row), created


async def list_documents(engine: AsyncEngine) -> list[DocumentRow]:
    async with engine.connect() as conn:
        rows = await conn.execute(
            select(*_DOCUMENT_COLUMNS).order_by(documents.c.created_at.desc())
        )
        return [_document(r) for r in rows]


async def get_document(engine: AsyncEngine, document_id: str) -> DocumentRow | None:
    async with engine.connect() as conn:
        row = (
            await conn.execute(select(*_DOCUMENT_COLUMNS).where(documents.c.id == document_id))
        ).first()
        return _document(row) if row else None


async def get_raw_document(engine: AsyncEngine, document_id: str) -> RawDocument | None:
    async with engine.connect() as conn:
        row = (
            await conn.execute(
                select(documents.c.id, documents.c.filename, documents.c.content).where(
                    documents.c.id == document_id
                )
            )
        ).first()
        return RawDocument(row.id, row.filename, row.content) if row else None


async def delete_document(engine: AsyncEngine, document_id: str) -> bool:
    async with engine.begin() as conn:
        result = await conn.execute(delete(documents).where(documents.c.id == document_id))
        return result.rowcount > 0


_NO_PROGRESS = {"progress_stage": None, "progress_done": None, "progress_total": None}


async def set_progress(engine: AsyncEngine, job: Job, progress: Progress) -> None:
    """Record progress for the job's document; raises JobCancelled if the job is gone."""
    async with engine.begin() as conn:
        result = await conn.execute(
            update(documents)
            .where(documents.c.id == job.document_id, exists().where(_owned(job)))
            .values(
                progress_stage=progress.stage.value,
                progress_done=progress.done,
                progress_total=progress.total,
                updated_at=func.now(),
            )
        )
        if result.rowcount == 0:
            raise JobCancelled(job.id)


async def set_document_status(
    conn: AsyncConnection, document_id: str, status: DocumentStatus, **fields: Any
) -> None:
    await conn.execute(
        update(documents)
        .where(documents.c.id == document_id)
        .values(status=status, updated_at=func.now(), **fields)
    )


# ---------------------------------------------------------------- chunks


def make_save_chunks(engine: AsyncEngine, job: Job | None = None) -> SaveChunks:
    """Replace a document's chunks. With a job, only while this worker still owns it
    (the row lock keeps a concurrent delete from slipping in between)."""

    async def save(
        document_id: str, items: Sequence[Chunk], vectors: Sequence[Vector], model: str
    ) -> None:
        if len(items) != len(vectors):
            raise ValueError("one vector per chunk is required")
        async with engine.begin() as conn:
            if job is not None:
                locked = await conn.execute(select(jobs.c.id).where(_owned(job)).with_for_update())
                if locked.first() is None:
                    raise JobCancelled(job.id)
            await conn.execute(delete(chunks).where(chunks.c.document_id == document_id))
            if items:
                await conn.execute(
                    insert(chunks),
                    [
                        {
                            "document_id": c.document_id,
                            "ordinal": c.ordinal,
                            "page": c.page,
                            "text": c.text,
                            "context": c.context,
                            "embedding": list(v),
                            "embedding_model": model,
                        }
                        for c, v in zip(items, vectors, strict=True)
                    ],
                )

    return save


_HIT_COLUMNS = "c.id, c.document_id, d.filename, c.page, c.ordinal, c.text"
_SCOPE = (
    "d.status = 'ready' AND "
    "(CAST(:ids AS text[]) IS NULL OR c.document_id = ANY(CAST(:ids AS text[])))"
)


def _hits(rows: Any) -> list[Hit]:
    return [
        Hit(
            chunk_id=r.id,
            document_id=r.document_id,
            filename=r.filename,
            page=r.page,
            ordinal=r.ordinal,
            text=r.text,
            score=float(r.score),
        )
        for r in rows
    ]


def make_vector_search(engine: AsyncEngine, model: str) -> VectorSearch:
    sql = text(
        f"""
        SELECT {_HIT_COLUMNS}, 1 - (c.embedding <=> CAST(:q AS vector)) AS score
        FROM chunks c JOIN documents d ON d.id = c.document_id
        WHERE c.embedding_model = :model AND {_SCOPE}
        ORDER BY c.embedding <=> CAST(:q AS vector), c.id
        LIMIT :k
        """
    )

    async def search(vector: Vector, k: int, ids: Sequence[str] | None) -> list[Hit]:
        async with engine.begin() as conn:
            # The HNSW index filters after its scan; without this, scoping the search to a
            # few documents can return fewer than k rows once the index is used.
            await conn.execute(text("SET LOCAL hnsw.iterative_scan = strict_order"))
            rows = await conn.execute(
                sql,
                {
                    "q": vector_literal(vector),
                    "model": model,
                    "k": k,
                    "ids": list(ids) if ids else None,
                },
            )
            return _hits(rows)

    return search


def make_text_search(engine: AsyncEngine) -> TextSearch:
    sql = text(
        f"""
        SELECT {_HIT_COLUMNS}, ts_rank_cd(c.tsv, q) AS score
        FROM chunks c JOIN documents d ON d.id = c.document_id,
             to_tsquery('simple', :q) AS q
        WHERE c.tsv @@ q AND {_SCOPE}
        ORDER BY score DESC, c.id
        LIMIT :k
        """
    )

    async def search(tsquery: str, k: int, ids: Sequence[str] | None) -> list[Hit]:
        async with engine.connect() as conn:
            rows = await conn.execute(
                sql, {"q": tsquery, "k": k, "ids": list(ids) if ids else None}
            )
            return _hits(rows)

    return search


# ---------------------------------------------------------------- jobs


def _lease(seconds: float) -> Any:
    return func.now() + func.make_interval(0, 0, 0, 0, 0, 0, seconds)


async def claim_job(engine: AsyncEngine, worker: str, *, lease_seconds: float) -> Job | None:
    """Take the oldest pending job. SKIP LOCKED lets several workers share the queue."""
    async with engine.begin() as conn:
        row = (
            await conn.execute(
                text(
                    """
                    UPDATE jobs SET status = 'running', attempts = attempts + 1,
                        worker_id = :worker, lease_until = now() + make_interval(secs => :lease),
                        updated_at = now()
                    WHERE id = (
                        SELECT id FROM jobs WHERE status = 'pending'
                        ORDER BY id FOR UPDATE SKIP LOCKED LIMIT 1
                    )
                    RETURNING id, document_id, options, attempts
                    """
                ),
                {"worker": worker, "lease": lease_seconds},
            )
        ).first()
        if row is None:
            return None
        await set_document_status(conn, row.document_id, DocumentStatus.PROCESSING, **_NO_PROGRESS)
        return Job(row.id, row.document_id, dict(row.options), row.attempts, worker)


async def renew_lease(engine: AsyncEngine, job: Job, *, lease_seconds: float) -> bool:
    """Extend the lease; False once the job is no longer this worker's."""
    async with engine.begin() as conn:
        renewed = await conn.execute(
            update(jobs)
            .where(_owned(job))
            .values(lease_until=_lease(lease_seconds))
            .returning(jobs.c.id)
        )
        return renewed.first() is not None


async def finish_job(
    engine: AsyncEngine, job: Job, *, error: str | None, document_fields: dict[str, Any]
) -> None:
    """Close the job and update its document, unless the job is no longer this worker's."""
    async with engine.begin() as conn:
        closed = await conn.execute(
            update(jobs)
            .where(_owned(job))
            .values(
                status="failed" if error else "done",
                error=error,
                lease_until=None,
                updated_at=func.now(),
            )
            .returning(jobs.c.id)
        )
        if closed.first() is None:
            return
        status = DocumentStatus.FAILED if error else DocumentStatus.READY
        await set_document_status(
            conn, job.document_id, status, error=error, **_NO_PROGRESS, **document_fields
        )


async def _queue(conn: AsyncConnection, document_id: str, options: dict[str, Any]) -> None:
    await conn.execute(
        update(jobs)
        .where(jobs.c.document_id == document_id, jobs.c.status.in_(["pending", "running"]))
        .values(status="cancelled", lease_until=None, updated_at=func.now())
    )
    await conn.execute(insert(jobs).values(document_id=document_id, options=options))
    await set_document_status(conn, document_id, DocumentStatus.QUEUED, error=None)


async def requeue(engine: AsyncEngine, document_id: str, options: dict[str, Any]) -> None:
    """Queue a fresh ingestion; any open job for the document is cancelled (and stops)."""
    async with engine.begin() as conn:
        await _queue(conn, document_id, options)


async def recover_expired_jobs(engine: AsyncEngine, *, max_attempts: int) -> int:
    """Jobs whose worker stopped renewing the lease (crash, kill, lost connection) go back
    to the queue, or fail once they have used all their attempts."""
    expired = (jobs.c.status == "running") & (
        jobs.c.lease_until.is_(None) | (jobs.c.lease_until < func.now())
    )
    give_up = f"Stopped {max_attempts} times while processing this document; not retrying."
    async with engine.begin() as conn:
        retried = (
            (
                await conn.execute(
                    update(jobs)
                    .where(expired, jobs.c.attempts < max_attempts)
                    .values(
                        status="pending", worker_id=None, lease_until=None, updated_at=func.now()
                    )
                    .returning(jobs.c.document_id)
                )
            )
            .scalars()
            .all()
        )
        for document_id in retried:
            await set_document_status(conn, document_id, DocumentStatus.QUEUED, **_NO_PROGRESS)
        failed = (
            (
                await conn.execute(
                    update(jobs)
                    .where(expired)
                    .values(status="failed", error=give_up, lease_until=None, updated_at=func.now())
                    .returning(jobs.c.document_id)
                )
            )
            .scalars()
            .all()
        )
        for document_id in failed:
            await set_document_status(
                conn, document_id, DocumentStatus.FAILED, error=give_up, **_NO_PROGRESS
            )
    return len(retried) + len(failed)


async def requeue_stale_embeddings(engine: AsyncEngine, model: str) -> int:
    """Documents embedded with another model are invisible to vector search; ingest them
    again with the options of their last job."""
    async with engine.begin() as conn:
        rows = await conn.execute(
            text(
                """
                SELECT DISTINCT ON (j.document_id) j.document_id, j.options
                FROM jobs j JOIN documents d ON d.id = j.document_id
                WHERE d.status = 'ready' AND EXISTS (
                    SELECT 1 FROM chunks c
                    WHERE c.document_id = d.id AND c.embedding_model <> :model
                )
                ORDER BY j.document_id, j.id DESC
                """
            ),
            {"model": model},
        )
        stale = [(row.document_id, dict(row.options)) for row in rows]
        for document_id, options in stale:
            await _queue(conn, document_id, options)
    return len(stale)
