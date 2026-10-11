"""HTTP request/response models, and pure converters from domain values."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

from raglab.db.repository import DocumentRow
from raglab.domain import Citation, Hit, Progress, Prompt, Stage, Turn


class ProgressOut(BaseModel):
    stage: str
    done: int
    total: int  # 0 when the total is not known yet (one AI call reading a scanned PDF)
    percent: int


class DocumentOut(BaseModel):
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
    progress: ProgressOut | None
    created_at: datetime


class HitOut(BaseModel):
    chunk_id: int
    document_id: str
    filename: str
    page: int
    text: str
    score: float


class CitationOut(BaseModel):
    number: int
    hit: HitOut


class ProviderOut(BaseModel):
    id: str
    label: str
    kind: Literal["cli", "api"]


class ProvidersOut(BaseModel):
    generators: list[ProviderOut]
    transcribers: list[str]
    embedding_model: str
    default_generator: str | None


class TurnIn(BaseModel):
    role: Literal["user", "assistant"]
    text: str = Field(max_length=20_000)


class SearchIn(BaseModel):
    query: str = Field(min_length=1, max_length=2_000)
    k: int = Field(default=6, ge=1, le=20)
    document_ids: list[str] | None = None


class ContextIn(SearchIn):
    history: list[TurnIn] = Field(default_factory=list, max_length=20)


class AskIn(ContextIn):
    provider: str


class ContextOut(BaseModel):
    """Sources plus the exact prompt, for generators that run in the browser (Gemini Nano)."""

    hits: list[HitOut]
    system: str
    user: str


class ReprocessIn(BaseModel):
    contextualize_with: str | None = None
    ai_extraction: str | None = None


def progress_out(row: DocumentRow) -> ProgressOut | None:
    if row.status != "processing" or row.progress_stage is None:
        return None
    progress = Progress(Stage(row.progress_stage), row.progress_done or 0, row.progress_total or 0)
    return ProgressOut(
        stage=progress.stage.value,
        done=progress.done,
        total=progress.total,
        percent=progress.percent,
    )


def document_out(row: DocumentRow) -> DocumentOut:
    return DocumentOut(
        id=row.id,
        filename=row.filename,
        size_bytes=row.size_bytes,
        status=row.status,
        pages=row.pages,
        chunks=row.chunks,
        extractor=row.extractor,
        contextualized=row.contextualized,
        warnings=row.warnings,
        error=row.error,
        progress=progress_out(row),
        created_at=row.created_at,
    )


def hit_out(hit: Hit) -> HitOut:
    return HitOut(
        chunk_id=hit.chunk_id,
        document_id=hit.document_id,
        filename=hit.filename,
        page=hit.page,
        text=hit.text,
        score=hit.score,
    )


def citation_out(citation: Citation) -> CitationOut:
    return CitationOut(number=citation.number, hit=hit_out(citation.hit))


def context_out(hits: tuple[Hit, ...], prompt: Prompt) -> ContextOut:
    return ContextOut(hits=[hit_out(h) for h in hits], system=prompt.system, user=prompt.user)


def turns(history: list[TurnIn]) -> tuple[Turn, ...]:
    return tuple(Turn(role=t.role, text=t.text) for t in history)
