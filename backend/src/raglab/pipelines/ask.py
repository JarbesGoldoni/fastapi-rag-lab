"""Search and answer pipelines."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass

from raglab.core.prompts import answer_prompt, citations
from raglab.core.retrieval import fuse_rrf, within_budget
from raglab.core.text import or_tsquery
from raglab.domain import Citation, Hit, Restart, Turn
from raglab.ports import Embedder, Generator, TextSearch, VectorSearch

NO_SOURCES = "I couldn't find anything about that in your documents."


async def hybrid_search(
    query: str,
    *,
    embedder: Embedder,
    vector_search: VectorSearch,
    text_search: TextSearch,
    k: int = 6,
    candidates: int = 20,
    document_ids: Sequence[str] | None = None,
) -> tuple[Hit, ...]:
    """Semantic + keyword search run concurrently, fused by rank."""
    (vector,) = await embedder.query([query])
    tsquery = or_tsquery(query)
    async with asyncio.TaskGroup() as group:
        by_meaning = group.create_task(vector_search(vector, candidates, document_ids))
        by_words = (
            group.create_task(text_search(tsquery, candidates, document_ids)) if tsquery else None
        )
    rankings = [by_meaning.result()] + ([by_words.result()] if by_words else [])
    return fuse_rrf(rankings, limit=k)


@dataclass(frozen=True, slots=True)
class Sources:
    hits: tuple[Hit, ...]


@dataclass(frozen=True, slots=True)
class Delta:
    text: str


@dataclass(frozen=True, slots=True)
class Reset:
    """Discard the text streamed so far; the model restarted its answer."""


@dataclass(frozen=True, slots=True)
class Done:
    answer: str
    citations: tuple[Citation, ...]


type AskEvent = Sources | Delta | Reset | Done


async def ask(
    question: str,
    hits: Sequence[Hit],
    *,
    generator: Generator,
    history: Sequence[Turn] = (),
    max_context_chars: int = 9_000,
) -> AsyncIterator[AskEvent]:
    selected = within_budget(hits, max_chars=max_context_chars)
    yield Sources(selected)
    if not selected:
        yield Delta(NO_SOURCES)
        yield Done(NO_SOURCES, ())
        return
    parts: list[str] = []
    async for delta in generator.stream(answer_prompt(question, selected, history)):
        if isinstance(delta, Restart):
            if parts:
                parts.clear()
                yield Reset()
            continue
        parts.append(delta)
        yield Delta(delta)
    answer = "".join(parts)
    yield Done(answer, citations(answer, selected))
