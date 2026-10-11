"""Deterministic stand-ins for the effectful ports. No network, no models, no database."""

from __future__ import annotations

import hashlib
import math
from collections.abc import AsyncIterator, Sequence

import pymupdf

from raglab.core.text import keywords
from raglab.domain import Chunk, Page, Progress, Prompt, RawDocument, Stage, Vector
from raglab.ports import Embedder, Extractor, Generator, Report

DIMENSIONS = 384


def hash_vector(text: str, dimensions: int = DIMENSIONS) -> Vector:
    """Bag-of-words hashed into a fixed-size unit vector: shared words -> higher cosine."""
    values = [0.0] * dimensions
    for word in keywords(text, min_length=2):
        slot = int(hashlib.blake2b(word.encode(), digest_size=4).hexdigest(), 16) % dimensions
        values[slot] += 1.0
    norm = math.sqrt(sum(v * v for v in values)) or 1.0
    return tuple(v / norm for v in values)


async def _embed(texts: Sequence[str]) -> list[Vector]:
    return [hash_vector(t) for t in texts]


hash_embedder = Embedder(model="test-hash", dimensions=DIMENSIONS, documents=_embed, query=_embed)


def fixed_extractor(pages: Sequence[Page], id: str = "fixed") -> Extractor:
    async def extract(_: RawDocument, report: Report) -> tuple[Page, ...]:
        for i in range(len(pages) + 1):
            await report(Progress(Stage.READING, i, len(pages)))
        return tuple(pages)

    return Extractor(id=id, extract=extract)


def scripted_generator(*replies: str, id: str = "fake") -> tuple[Generator, list[Prompt]]:
    """Streams each reply word by word, in order; records the prompts it received."""
    received: list[Prompt] = []

    async def stream(prompt: Prompt) -> AsyncIterator[str]:
        received.append(prompt)
        reply = replies[min(len(received), len(replies)) - 1]
        for i, word in enumerate(reply.split(" ")):
            yield word if i == 0 else " " + word

    return Generator(id=id, label="Fake", kind="api", stream=stream), received


def failing_generator(message: str = "boom") -> Generator:
    async def stream(_: Prompt) -> AsyncIterator[str]:
        raise RuntimeError(message)
        yield ""  # pragma: no cover - makes this an async generator

    return Generator(id="failing", label="Failing", kind="api", stream=stream)


class MemoryStore:
    def __init__(self) -> None:
        self.saved: dict[str, tuple[list[Chunk], list[Vector], str]] = {}

    async def save(
        self, document_id: str, chunks: Sequence[Chunk], vectors: Sequence[Vector], model: str
    ) -> None:
        self.saved[document_id] = (list(chunks), list(vectors), model)


def make_pdf(pages: Sequence[str]) -> bytes:
    """A real PDF with a text layer, one string per page."""
    with pymupdf.open() as pdf:
        for text in pages:
            page = pdf.new_page()
            page.insert_textbox(pymupdf.Rect(50, 50, 550, 800), text, fontsize=11)
        return pdf.tobytes()


class ProgressLog:
    def __init__(self) -> None:
        self.updates: list[Progress] = []

    async def report(self, progress: Progress) -> None:
        self.updates.append(progress)

    def stages(self) -> list[str]:
        return list(dict.fromkeys(p.stage.value for p in self.updates))

    def last(self, stage: Stage) -> Progress:
        return [p for p in self.updates if p.stage == stage][-1]
