from datetime import UTC, datetime

import pytest

from raglab.adapters.pdf import extract_text_layer
from raglab.api.schemas import progress_out
from raglab.db.repository import DocumentRow
from raglab.domain import ChunkConfig, Page, Progress, RawDocument, Stage
from raglab.pipelines.concurrency import throttled
from raglab.pipelines.ingest import ingest
from raglab.ports import IngestOptions
from tests.fakes import (
    MemoryStore,
    ProgressLog,
    fixed_extractor,
    hash_embedder,
    make_pdf,
    scripted_generator,
)

pytestmark = pytest.mark.anyio


def test_percent_is_floored_and_safe_for_unknown_totals() -> None:
    assert Progress(Stage.EMBEDDING, 128, 340).percent == 37
    assert Progress(Stage.EMBEDDING, 340, 340).percent == 100
    assert Progress(Stage.READING, 0, 0).percent == 0


async def test_ingest_reports_every_stage_with_real_counts() -> None:
    log = ProgressLog()
    pages = [
        Page(i, f"Page {i} talks about refunds and settlement windows. " * 30) for i in range(1, 6)
    ]
    result = await ingest(
        RawDocument("d", "f.pdf", b"%PDF"),
        IngestOptions(),
        extractor=fixed_extractor(pages),
        embedder=hash_embedder,
        save=MemoryStore().save,
        cfg=ChunkConfig(size=300, overlap=0),
        embed_batch=4,
        report=log.report,
    )
    assert log.stages() == ["reading", "embedding", "saving"]
    assert log.last(Stage.READING) == Progress(Stage.READING, 5, 5)
    embedding = [p for p in log.updates if p.stage == Stage.EMBEDDING]
    assert embedding[0].done == 0 and embedding[-1].done == embedding[-1].total == result.chunks
    assert [p.done for p in embedding] == sorted(p.done for p in embedding)  # never goes back
    assert len(embedding) > 2  # one update per batch, not just start and end
    assert log.updates[-1] == Progress(Stage.SAVING, result.chunks, result.chunks)


async def test_contextualize_reports_passages_done() -> None:
    log = ProgressLog()
    generator, _ = scripted_generator('["ctx"]')
    await ingest(
        RawDocument("d", "f.pdf", b"%PDF"),
        IngestOptions(contextualize_with=generator),
        extractor=fixed_extractor([Page(1, "Refunds take fourteen days.")]),
        embedder=hash_embedder,
        save=MemoryStore().save,
        report=log.report,
    )
    assert log.stages() == ["reading", "context", "embedding", "saving"]
    assert log.last(Stage.CONTEXT) == Progress(Stage.CONTEXT, 1, 1)


async def test_pdf_extraction_reports_pages_read() -> None:
    log = ProgressLog()
    content = make_pdf([f"page {i}" for i in range(20)])
    pages = await extract_text_layer(RawDocument("d", "f.pdf", content), log.report)
    assert len(pages) == 20
    assert [p.done for p in log.updates] == [0, 8, 16, 20]
    assert all(p.total == 20 for p in log.updates)


async def test_throttle_keeps_first_and_last_and_spaces_the_rest() -> None:
    sent = ProgressLog()
    now = [0.0]
    report = throttled(sent.report, interval=1.0, clock=lambda: now[0])
    for done in range(11):  # 11 updates within 0.5 s
        now[0] = done * 0.05
        await report(Progress(Stage.EMBEDDING, done, 10))
    now[0] = 0.6
    await report(Progress(Stage.SAVING, 0, 10))
    assert [(p.stage.value, p.done) for p in sent.updates] == [
        ("embedding", 0),
        ("embedding", 10),
        ("saving", 0),
    ]


def _row(status: str, stage: str | None, done: int | None, total: int | None) -> DocumentRow:
    return DocumentRow(
        "id", "f.pdf", 1, status, None, None, None, None, [], None, stage, done, total,
        datetime.now(UTC),
    )  # fmt: skip


def test_progress_is_exposed_only_while_processing() -> None:
    out = progress_out(_row("processing", "embedding", 128, 340))
    assert out is not None and (out.stage, out.done, out.total, out.percent) == (
        "embedding",
        128,
        340,
        37,
    )
    assert progress_out(_row("ready", "saving", 340, 340)) is None
    assert progress_out(_row("processing", None, None, None)) is None
