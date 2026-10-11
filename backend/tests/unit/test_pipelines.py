import asyncio
from collections.abc import AsyncIterator, Sequence

import pytest

from raglab.domain import RESTART, ChunkConfig, Hit, Page, Prompt, RawDocument, Restart, Vector
from raglab.pipelines.ask import NO_SOURCES, Delta, Done, Reset, Sources, ask, hybrid_search
from raglab.pipelines.concurrency import map_batched
from raglab.pipelines.ingest import IngestError, ingest
from raglab.ports import Generator, IngestOptions
from tests.fakes import (
    MemoryStore,
    failing_generator,
    fixed_extractor,
    hash_embedder,
    scripted_generator,
)

pytestmark = pytest.mark.anyio

DOC = RawDocument(id="doc1", filename="handbook.pdf", content=b"%PDF")
TEXT = "Refunds are processed within fourteen days of the request. " * 20


async def test_ingest_chunks_embeds_and_saves() -> None:
    store = MemoryStore()
    result = await ingest(
        DOC,
        IngestOptions(),
        extractor=fixed_extractor([Page(1, TEXT), Page(2, "Support hours are nine to six.")]),
        embedder=hash_embedder,
        save=store.save,
        cfg=ChunkConfig(size=300, overlap=50),
    )
    chunks, vectors, model = store.saved["doc1"]
    assert result.chunks == len(chunks) == len(vectors) > 2
    assert result.pages == 2 and result.extractor == "fixed" and result.warnings == ()
    assert model == "test-hash"
    assert chunks[-1].page == 2


async def test_ingest_falls_back_to_ai_extraction_for_scanned_pdfs() -> None:
    store = MemoryStore()
    result = await ingest(
        DOC,
        IngestOptions(ai_extractor=fixed_extractor([Page(1, TEXT)], id="ai")),
        extractor=fixed_extractor([Page(1, "  ")]),
        embedder=hash_embedder,
        save=store.save,
    )
    assert result.extractor == "ai"
    assert store.saved["doc1"][0][0].text.startswith("Refunds")


async def test_ingest_without_text_or_ai_fails_with_a_warning_path() -> None:
    with pytest.raises(IngestError):
        await ingest(
            DOC,
            IngestOptions(),
            extractor=fixed_extractor([]),
            embedder=hash_embedder,
            save=MemoryStore().save,
        )


async def test_ingest_contextualizes_chunks_and_embeds_the_context() -> None:
    store = MemoryStore()
    generator, prompts = scripted_generator('["Refund policy section."]')
    result = await ingest(
        DOC,
        IngestOptions(contextualize_with=generator),
        extractor=fixed_extractor([Page(1, "Refunds take fourteen days.")]),
        embedder=hash_embedder,
        save=store.save,
    )
    (chunk,), _, _ = store.saved["doc1"]
    assert chunk.context == "Refund policy section."
    assert result.contextualized and len(prompts) == 1


async def test_contextualize_failure_keeps_chunks_and_warns() -> None:
    store = MemoryStore()
    result = await ingest(
        DOC,
        IngestOptions(contextualize_with=failing_generator()),
        extractor=fixed_extractor([Page(1, "Refunds take fourteen days.")]),
        embedder=hash_embedder,
        save=store.save,
    )
    assert store.saved["doc1"][0][0].context == ""
    assert result.warnings == ("1 of 1 chunks have no context",)


async def test_map_batched_keeps_order_and_bounds_concurrency() -> None:
    in_flight = peak = 0

    async def double(part: Sequence[int]) -> list[int]:
        nonlocal in_flight, peak
        in_flight += 1
        peak = max(peak, in_flight)
        await asyncio.sleep(0.01 * (5 - part[0] % 5))  # later batches finish first
        in_flight -= 1
        return [x * 2 for x in part]

    result = await map_batched(double, list(range(20)), batch=3, concurrency=2)
    assert result == [x * 2 for x in range(20)]
    assert peak == 2


def _hit(chunk_id: int) -> Hit:
    return Hit(chunk_id, "doc1", "handbook.pdf", 1, chunk_id, f"chunk {chunk_id}", 0.0)


async def test_hybrid_search_runs_both_searches_and_fuses() -> None:
    calls: list[str] = []

    async def vector_search(v: Vector, k: int, ids: Sequence[str] | None) -> list[Hit]:
        calls.append("vector")
        return [_hit(1), _hit(2)]

    async def text_search(q: str, k: int, ids: Sequence[str] | None) -> list[Hit]:
        calls.append(f"text:{q}")
        return [_hit(2), _hit(3)]

    hits = await hybrid_search(
        "refund window",
        embedder=hash_embedder,
        vector_search=vector_search,
        text_search=text_search,
        k=3,
    )
    assert [h.chunk_id for h in hits] == [2, 1, 3]
    assert sorted(calls) == ["text:refund | window", "vector"]


async def test_hybrid_search_skips_text_search_without_keywords() -> None:
    async def vector_search(v: Vector, k: int, ids: Sequence[str] | None) -> list[Hit]:
        return [_hit(1)]

    async def text_search(q: str, k: int, ids: Sequence[str] | None) -> list[Hit]:
        raise AssertionError("should not run")

    hits = await hybrid_search(
        "what is it?", embedder=hash_embedder, vector_search=vector_search, text_search=text_search
    )
    assert [h.chunk_id for h in hits] == [1]


async def test_ask_streams_sources_deltas_and_citations() -> None:
    generator, prompts = scripted_generator("Fourteen days [1].")
    events = [e async for e in ask("How long?", [_hit(1), _hit(2)], generator=generator)]
    assert isinstance(events[0], Sources) and len(events[0].hits) == 2
    assert "".join(e.text for e in events if isinstance(e, Delta)) == "Fourteen days [1]."
    done = events[-1]
    assert isinstance(done, Done)
    assert [c.hit.chunk_id for c in done.citations] == [1]
    assert "Question: How long?" in prompts[0].user


async def test_ask_without_sources_does_not_call_the_model() -> None:
    events = [e async for e in ask("?", [], generator=failing_generator())]
    assert events[1] == Delta(NO_SOURCES)
    assert isinstance(events[-1], Done) and events[-1].citations == ()


async def test_ask_discards_text_when_the_model_restarts() -> None:
    async def stream(_: Prompt) -> AsyncIterator[str | Restart]:
        for item in (RESTART, "Refunds are cre", RESTART, "Up to 180 days [1]."):
            yield item

    generator = Generator(id="g", label="G", kind="cli", stream=stream)
    events = [e async for e in ask("q", [_hit(1)], generator=generator)]
    assert sum(isinstance(e, Reset) for e in events) == 1  # the first restart had no text
    done = events[-1]
    assert isinstance(done, Done) and done.answer == "Up to 180 days [1]."
