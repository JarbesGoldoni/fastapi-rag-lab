"""Deterministic chunking. Chunks never cross a page, so every citation points at one page."""

from __future__ import annotations

from collections.abc import Iterator, Sequence

from raglab.domain import Chunk, ChunkConfig, Page


def windows(words: Sequence[str], *, size: int, overlap: int) -> Iterator[tuple[int, int]]:
    """Yield [start, end) word ranges whose joined length is <= size.

    Consecutive windows share up to `overlap` characters. A single word longer than
    `size` becomes its own window, so no text is ever dropped.
    """
    if size <= 0:
        raise ValueError("size must be positive")
    if not 0 <= overlap < size:
        raise ValueError("overlap must be >= 0 and smaller than size")

    start = 0
    while start < len(words):
        end, length = start, 0
        while end < len(words):
            added = len(words[end]) + (1 if end > start else 0)
            if end > start and length + added > size:
                break
            length += added
            end += 1
        yield start, end
        if end >= len(words):
            return
        back, shared = end, 0
        while back - 1 > start and shared + len(words[back - 1]) + 1 <= overlap:
            back -= 1
            shared += len(words[back]) + 1
        start = back


def chunk_page(page: Page, *, cfg: ChunkConfig) -> tuple[str, ...]:
    words = page.text.split()
    return tuple(
        " ".join(words[s:e]) for s, e in windows(words, size=cfg.size, overlap=cfg.overlap)
    )


def chunk_pages(document_id: str, pages: Sequence[Page], *, cfg: ChunkConfig) -> tuple[Chunk, ...]:
    texts = ((page.number, text) for page in pages for text in chunk_page(page, cfg=cfg))
    return tuple(
        Chunk(document_id=document_id, ordinal=i, page=number, text=text)
        for i, (number, text) in enumerate(texts)
    )


def with_contexts(chunks: Sequence[Chunk], contexts: Sequence[str]) -> tuple[Chunk, ...]:
    """Attach situating contexts by position. Missing contexts leave a chunk unchanged."""
    return tuple(
        Chunk(c.document_id, c.ordinal, c.page, c.text, contexts[i].strip())
        if i < len(contexts) and contexts[i].strip()
        else c
        for i, c in enumerate(chunks)
    )
