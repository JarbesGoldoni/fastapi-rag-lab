"""End to end over HTTP with a real Postgres + pgvector; only the models are faked.

Run with TEST_DATABASE_URL=postgresql+asyncpg://localhost:5432/raglab_test (migrated).
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import os
from collections.abc import AsyncIterator
from typing import Any

import httpx
import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

from raglab.adapters.registry import Providers
from raglab.main import create_app
from raglab.settings import Settings
from raglab.wiring import Container, build
from tests.fakes import hash_embedder, make_pdf, scripted_generator

DATABASE_URL = os.environ.get("TEST_DATABASE_URL")

pytestmark = [
    pytest.mark.anyio,
    pytest.mark.integration,
    pytest.mark.skipif(not DATABASE_URL, reason="TEST_DATABASE_URL is not set"),
]

HANDBOOK = [
    "Refund policy. Customers can request a refund within thirty days of purchase. "
    "Refunds are paid back to the original card within fourteen business days.",
    "Support hours. The support team answers chats from nine to six, Monday to Friday. "
    "Urgent payment incidents are handled by the on-call engineer at any time.",
]


@pytest.fixture
async def client() -> AsyncIterator[httpx.AsyncClient]:
    assert DATABASE_URL
    engine = create_async_engine(DATABASE_URL)
    async with engine.begin() as conn:
        await conn.execute(text("TRUNCATE documents, chunks, jobs RESTART IDENTITY CASCADE"))
    await engine.dispose()

    generator, _ = scripted_generator("Within thirty days [1].", id="fake")
    settings = Settings(
        database_url=DATABASE_URL,
        worker_poll_seconds=0.05,
        cors_origins=["http://localhost:5173"],
        allowed_hosts=["test"],
    )

    async def build_container(s: Settings) -> Container:
        container = await build(s, embedder=hash_embedder)
        return dataclasses.replace(
            container, providers=Providers(generators={"fake": generator}, transcribers={})
        )

    app = create_app(settings, build_container=build_container)
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as http:
            yield http


async def upload(
    client: httpx.AsyncClient, pages: list[str], name: str = "handbook.pdf"
) -> dict[str, Any]:
    files = {"file": (name, make_pdf(pages), "application/pdf")}
    response = await client.post("/api/documents", files=files)
    assert response.status_code in (200, 202), response.text
    return response.json()


async def wait_ready(client: httpx.AsyncClient, doc_id: str) -> dict[str, Any]:
    async with asyncio.timeout(15):
        while True:
            doc = (await client.get(f"/api/documents/{doc_id}")).json()
            if doc["status"] in ("ready", "failed"):
                return doc
            await asyncio.sleep(0.05)


def parse_sse(body: str) -> list[tuple[str, object]]:
    events = []
    for block in body.strip().split("\n\n"):
        lines = dict(line.split(": ", 1) for line in block.splitlines())
        events.append((lines["event"], json.loads(lines["data"])))
    return events


async def test_upload_ingest_search_and_ask(client: httpx.AsyncClient) -> None:
    doc = await upload(client, HANDBOOK)
    assert doc["status"] == "queued"

    ready = await wait_ready(client, doc["id"])
    assert ready["status"] == "ready", ready["error"]
    assert ready["pages"] == 2 and ready["chunks"] >= 2 and ready["extractor"] == "pymupdf"

    hits = (await client.post("/api/search", json={"query": "refund within days"})).json()
    assert hits[0]["page"] == 1 and "thirty days" in hits[0]["text"]

    response = await client.post(
        "/api/ask", json={"query": "How long do I have to ask for a refund?", "provider": "fake"}
    )
    assert response.headers["content-type"].startswith("text/event-stream")
    events = parse_sse(response.text)
    assert next(name for name, _ in events) == "sources"
    assert events[-1][0] == "done"
    done = events[-1][1]
    assert isinstance(done, dict)
    assert done["answer"] == "Within thirty days [1]."
    assert done["citations"][0]["hit"]["page"] == 1


async def test_same_file_twice_is_idempotent(client: httpx.AsyncClient) -> None:
    content = make_pdf(HANDBOOK)  # same bytes; make_pdf stamps a creation time
    first = await client.post("/api/documents", files={"file": ("a.pdf", content)})
    second = await client.post("/api/documents", files={"file": ("copy.pdf", content)})
    assert (first.status_code, second.status_code) == (202, 200)
    assert second.json()["id"] == first.json()["id"]
    assert len((await client.get("/api/documents")).json()) == 1


async def test_search_can_be_scoped_to_documents(client: httpx.AsyncClient) -> None:
    a = await upload(client, HANDBOOK, "a.pdf")
    b = await upload(client, ["Kafka consumers commit offsets after processing a record."], "b.pdf")
    await wait_ready(client, a["id"])
    await wait_ready(client, b["id"])
    scoped = await client.post(
        "/api/search", json={"query": "refund kafka", "document_ids": [b["id"]]}
    )
    assert {h["document_id"] for h in scoped.json()} == {b["id"]}


async def test_rejects_non_pdf_and_unknown_provider(client: httpx.AsyncClient) -> None:
    bad = await client.post("/api/documents", files={"file": ("x.txt", b"hello", "text/plain")})
    assert bad.status_code == 415
    ask = await client.post("/api/ask", json={"query": "hi", "provider": "nope"})
    assert ask.status_code == 422


async def test_delete_removes_document_and_chunks(client: httpx.AsyncClient) -> None:
    doc = await upload(client, HANDBOOK)
    await wait_ready(client, doc["id"])
    assert (await client.delete(f"/api/documents/{doc['id']}")).status_code == 204
    assert (await client.get(f"/api/documents/{doc['id']}")).status_code == 404
    assert (await client.post("/api/search", json={"query": "refund"})).json() == []


async def test_context_endpoint_returns_prompt_for_browser_models(
    client: httpx.AsyncClient,
) -> None:
    doc = await upload(client, HANDBOOK)
    await wait_ready(client, doc["id"])
    body = (await client.post("/api/context", json={"query": "support hours"})).json()
    assert body["hits"] and "[1]" in body["user"] and body["system"]


async def test_a_deleted_documents_job_cannot_write_to_its_reupload() -> None:
    """Same bytes -> same id. A job still running for the deleted copy must not touch
    the new row: progress raises JobCancelled, saving is refused, finishing is a no-op."""
    from raglab.db import repository as repo
    from raglab.domain import Chunk, Progress, RawDocument, Stage

    assert DATABASE_URL
    engine = create_async_engine(DATABASE_URL)
    try:
        async with engine.begin() as conn:
            await conn.execute(text("TRUNCATE documents, chunks, jobs RESTART IDENTITY CASCADE"))
        doc = RawDocument("same-id", "a.pdf", b"%PDF-1")
        await repo.create_document(engine, doc, {})
        old_job = await repo.claim_job(engine, "w1", lease_seconds=30)
        assert old_job is not None

        await repo.delete_document(engine, doc.id)
        await repo.create_document(engine, doc, {})  # re-upload, new pending job

        with pytest.raises(repo.JobCancelled):
            await repo.set_progress(engine, old_job, Progress(Stage.EMBEDDING, 5, 10))
        save = repo.make_save_chunks(engine, old_job)
        with pytest.raises(repo.JobCancelled):
            await save(doc.id, [Chunk(doc.id, 0, 1, "x")], [tuple([0.0] * 384)], "m")
        await repo.finish_job(engine, old_job, error=None, document_fields={"pages": 99})

        row = await repo.get_document(engine, doc.id)
        assert row is not None
        assert (row.status, row.pages, row.progress_stage) == ("queued", None, None)
    finally:
        await engine.dispose()


async def test_requeue_cancels_the_running_job() -> None:
    from raglab.db import repository as repo
    from raglab.domain import Progress, RawDocument, Stage

    assert DATABASE_URL
    engine = create_async_engine(DATABASE_URL)
    try:
        async with engine.begin() as conn:
            await conn.execute(text("TRUNCATE documents, chunks, jobs RESTART IDENTITY CASCADE"))
        doc = RawDocument("doc", "a.pdf", b"%PDF-1")
        await repo.create_document(engine, doc, {})
        running = await repo.claim_job(engine, "w1", lease_seconds=30)
        assert running is not None
        await repo.requeue(engine, doc.id, {})
        with pytest.raises(repo.JobCancelled):
            await repo.set_progress(engine, running, Progress(Stage.READING, 1, 2))
        fresh = await repo.claim_job(engine, "w1", lease_seconds=30)
        assert fresh is not None and fresh.id != running.id
    finally:
        await engine.dispose()


# ---------------------------------------------------------------- queue and worker


async def fresh_container(**overrides: Any) -> Container:
    assert DATABASE_URL
    container = await build(
        Settings(database_url=DATABASE_URL, worker_poll_seconds=0.05, **overrides),
        embedder=hash_embedder,
    )
    async with container.engine.begin() as conn:
        await conn.execute(text("TRUNCATE documents, chunks, jobs RESTART IDENTITY CASCADE"))
    return container


async def test_a_live_lease_is_not_taken_over_by_another_worker() -> None:
    from raglab.db import repository as repo
    from raglab.domain import RawDocument

    c = await fresh_container()
    try:
        await repo.create_document(c.engine, RawDocument("d", "a.pdf", b"%PDF-1"), {})
        assert await repo.claim_job(c.engine, "a", lease_seconds=30)
        assert await repo.recover_expired_jobs(c.engine, max_attempts=3) == 0
        assert await repo.claim_job(c.engine, "b", lease_seconds=30) is None
    finally:
        await c.engine.dispose()


async def test_an_expired_lease_moves_the_job_and_locks_out_the_old_worker() -> None:
    from raglab.db import repository as repo
    from raglab.domain import Progress, RawDocument, Stage

    c = await fresh_container()
    try:
        await repo.create_document(c.engine, RawDocument("d", "a.pdf", b"%PDF-1"), {})
        stalled = await repo.claim_job(c.engine, "a", lease_seconds=0)
        assert stalled is not None
        assert await repo.recover_expired_jobs(c.engine, max_attempts=3) == 1
        taken = await repo.claim_job(c.engine, "b", lease_seconds=30)
        assert taken is not None and taken.id == stalled.id and taken.attempts == 2

        assert not await repo.renew_lease(c.engine, stalled, lease_seconds=30)
        with pytest.raises(repo.JobCancelled):
            await repo.set_progress(c.engine, stalled, Progress(Stage.READING, 1, 2))
        await repo.set_progress(c.engine, taken, Progress(Stage.READING, 1, 2))
    finally:
        await c.engine.dispose()


async def test_a_job_that_keeps_stopping_its_worker_fails_after_max_attempts() -> None:
    from raglab.db import repository as repo
    from raglab.domain import RawDocument

    c = await fresh_container()
    try:
        await repo.create_document(c.engine, RawDocument("d", "a.pdf", b"%PDF-1"), {})
        for _ in range(2):
            assert await repo.claim_job(c.engine, "w", lease_seconds=0)
            await repo.recover_expired_jobs(c.engine, max_attempts=2)
        row = await repo.get_document(c.engine, "d")
        assert row is not None and row.status == "failed" and "not retrying" in (row.error or "")
        assert await repo.claim_job(c.engine, "w", lease_seconds=30) is None
    finally:
        await c.engine.dispose()


async def test_deleting_a_document_stops_a_running_model_call_within_a_heartbeat() -> None:
    from raglab import worker
    from raglab.db import repository as repo
    from raglab.domain import Prompt, RawDocument
    from raglab.ports import Generator

    c = await fresh_container(worker_heartbeat_seconds=0.05)
    cancelled = asyncio.Event()

    async def slow(_: Prompt) -> AsyncIterator[str]:
        try:
            await asyncio.sleep(30)  # one long CLI call
        except asyncio.CancelledError:
            cancelled.set()
            raise
        yield "never"

    c = dataclasses.replace(
        c,
        providers=Providers(
            generators={"slow": Generator(id="slow", label="S", kind="cli", stream=slow)},
            transcribers={},
        ),
    )
    try:
        doc = RawDocument("d", "a.pdf", make_pdf(HANDBOOK))
        await repo.create_document(c.engine, doc, {"contextualize_with": "slow"})
        job = await repo.claim_job(c.engine, "w", lease_seconds=30)
        assert job is not None
        task = asyncio.create_task(worker.process(c, job))
        await asyncio.sleep(0.3)
        await repo.delete_document(c.engine, doc.id)
        async with asyncio.timeout(2):
            await task
        assert cancelled.is_set()
    finally:
        await c.engine.dispose()


async def test_the_worker_keeps_going_after_a_job_fails_outside_ingestion() -> None:
    from raglab import worker
    from raglab.db import repository as repo
    from raglab.domain import RawDocument

    c = await fresh_container()
    handled: list[str] = []
    both = asyncio.Event()

    async def handle(_: Container, job: repo.Job) -> None:
        handled.append(job.document_id)
        if len(handled) == 2:
            both.set()
        if job.document_id == "first":
            raise ConnectionError("database connection lost")

    try:
        for name in ("first", "second"):
            await repo.create_document(c.engine, RawDocument(name, "a.pdf", b"%PDF-1"), {})
        stop = asyncio.Event()
        task = asyncio.create_task(worker.run(c, stop, handle=handle, worker="w"))
        async with asyncio.timeout(5):
            await both.wait()
        assert handled == ["first", "second"]
        assert not task.done()
        stop.set()
        await task
    finally:
        await c.engine.dispose()


async def test_documents_embedded_with_another_model_are_ingested_again() -> None:
    from raglab.db import repository as repo
    from raglab.domain import Chunk, RawDocument

    c = await fresh_container()
    try:
        await repo.create_document(c.engine, RawDocument("d", "a.pdf", b"%PDF-1"), {"k": "v"})
        job = await repo.claim_job(c.engine, "w", lease_seconds=30)
        assert job is not None
        save = repo.make_save_chunks(c.engine, job)
        await save("d", [Chunk("d", 0, 1, "x")], [tuple([0.1] * 384)], "old-model")
        await repo.finish_job(c.engine, job, error=None, document_fields={})

        assert await repo.requeue_stale_embeddings(c.engine, "old-model") == 0
        assert await repo.requeue_stale_embeddings(c.engine, "new-model") == 1
        again = await repo.claim_job(c.engine, "w", lease_seconds=30)
        assert again is not None and again.options == {"k": "v"}
    finally:
        await c.engine.dispose()


# ---------------------------------------------------------------- browser protection


async def test_cross_origin_writes_and_foreign_hosts_are_refused(
    client: httpx.AsyncClient,
) -> None:
    files = {"file": ("a.pdf", make_pdf(HANDBOOK), "application/pdf")}
    foreign = await client.post(
        "/api/documents", files=files, headers={"Origin": "https://evil.example"}
    )
    assert foreign.status_code == 403
    assert (await client.get("/api/documents")).json() == []

    listed = await client.post(
        "/api/documents", files=files, headers={"Origin": "http://localhost:5173"}
    )
    assert listed.status_code == 202

    rebound = await client.get("/api/documents", headers={"Host": "evil.example"})
    assert rebound.status_code == 400


async def test_health_reports_the_embedded_worker(client: httpx.AsyncClient) -> None:
    assert (await client.get("/api/health")).json() == {"status": "ok", "worker": "running"}
