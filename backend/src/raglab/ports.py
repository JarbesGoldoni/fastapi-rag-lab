"""The effects the pipelines need, as small function types.

Pipelines receive these as arguments. Adapters implement them; tests pass fakes.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable, Coroutine, Sequence
from dataclasses import dataclass
from typing import Any, Literal

from raglab.domain import (
    Chunk,
    Hit,
    IngestResult,
    Page,
    Progress,
    Prompt,
    RawDocument,
    Restart,
    Vector,
)

type Async[T] = Coroutine[Any, Any, T]
type Report = Callable[[Progress], Async[None]]
type Extract = Callable[[RawDocument, Report], Async[tuple[Page, ...]]]
type EmbedTexts = Callable[[Sequence[str]], Async[list[Vector]]]
type Stream = Callable[[Prompt], AsyncIterator[str | Restart]]
type SaveChunks = Callable[[str, Sequence[Chunk], Sequence[Vector], str], Async[None]]
type VectorSearch = Callable[[Vector, int, Sequence[str] | None], Async[list[Hit]]]
type TextSearch = Callable[[str, int, Sequence[str] | None], Async[list[Hit]]]
type Ingest = Callable[[RawDocument, "IngestOptions"], Async[IngestResult]]


@dataclass(frozen=True, slots=True)
class Embedder:
    model: str
    dimensions: int
    documents: EmbedTexts
    query: EmbedTexts  # some models embed queries with a different instruction


@dataclass(frozen=True, slots=True)
class Generator:
    id: str
    label: str
    kind: Literal["cli", "api"]
    stream: Stream


@dataclass(frozen=True, slots=True)
class Extractor:
    id: str
    extract: Extract


@dataclass(frozen=True, slots=True)
class IngestOptions:
    contextualize_with: Generator | None = None
    ai_extractor: Extractor | None = None
