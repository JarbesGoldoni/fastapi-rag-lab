"""Immutable values shared by every layer. No behavior with side effects lives here."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Literal

type Vector = tuple[float, ...]


class DocumentStatus(StrEnum):
    QUEUED = "queued"
    PROCESSING = "processing"
    READY = "ready"
    FAILED = "failed"


class Stage(StrEnum):
    READING = "reading"
    CONTEXT = "context"
    EMBEDDING = "embedding"
    SAVING = "saving"


@dataclass(frozen=True, slots=True)
class Progress:
    """Measured progress of one ingestion stage: `done` of `total` units (pages or passages)."""

    stage: Stage
    done: int
    total: int

    @property
    def percent(self) -> int:
        return min(100, self.done * 100 // self.total) if self.total > 0 else 0


@dataclass(frozen=True, slots=True)
class RawDocument:
    id: str
    filename: str
    content: bytes


@dataclass(frozen=True, slots=True)
class Page:
    number: int  # 1-based, as a reader sees it
    text: str


@dataclass(frozen=True, slots=True)
class Chunk:
    document_id: str
    ordinal: int
    page: int
    text: str
    context: str = ""

    @property
    def embedding_text(self) -> str:
        """What gets embedded and full-text indexed: situating context first, then the text."""
        return f"{self.context}\n\n{self.text}" if self.context else self.text


@dataclass(frozen=True, slots=True)
class Hit:
    chunk_id: int
    document_id: str
    filename: str
    page: int
    ordinal: int
    text: str
    score: float


@dataclass(frozen=True, slots=True)
class Turn:
    role: Literal["user", "assistant"]
    text: str


@dataclass(frozen=True, slots=True)
class Prompt:
    system: str
    user: str


@dataclass(frozen=True, slots=True)
class Citation:
    number: int  # the [n] the model wrote
    hit: Hit


@dataclass(frozen=True, slots=True)
class Restart:
    """Yielded by a generator stream when the model restarts its answer (e.g. a retry).

    Everything streamed before it should be discarded.
    """


RESTART = Restart()


@dataclass(frozen=True, slots=True)
class ChunkConfig:
    size: int = 900
    overlap: int = 150


@dataclass(frozen=True, slots=True)
class IngestResult:
    document_id: str
    pages: int
    chunks: int
    extractor: str
    contextualized: bool = False
    warnings: tuple[str, ...] = field(default_factory=tuple)
