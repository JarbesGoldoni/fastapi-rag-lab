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
    settings = Settings(database_url=DATABASE_URL, worker_poll_seconds=0.05, cors_origins=[])

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
        old_job = await repo.claim_job(engine)
        assert old_job is not None

        await repo.delete_document(engine, doc.id)
        await repo.create_document(engine, doc, {})  # re-upload, new pending job

        with pytest.raises(repo.JobCancelled):
            await repo.set_progress(engine, old_job, Progress(Stage.EMBEDDING, 5, 10))
        save = repo.make_save_chunks(engine, old_job.id)
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
        running = await repo.claim_job(engine)
        assert running is not None
        await repo.requeue(engine, doc.id, {})
        with pytest.raises(repo.JobCancelled):
            await repo.set_progress(engine, running, Progress(Stage.READING, 1, 2))
        fresh = await repo.claim_job(engine)
        assert fresh is not None and fresh.id != running.id
    finally:
        await engine.dispose()
