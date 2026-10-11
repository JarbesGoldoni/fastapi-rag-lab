"""Pure ranking helpers for hybrid search."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace

from raglab.domain import Hit


def fuse_rrf(rankings: Sequence[Sequence[Hit]], *, k: int = 60, limit: int = 8) -> tuple[Hit, ...]:
    """Reciprocal rank fusion: score = sum(1 / (k + rank)) over every ranking a chunk appears in.

    Ranks matter, raw scores don't, so cosine similarity and ts_rank can be combined safely.
    Ties break on chunk_id to keep the output deterministic.
    """
    scores: dict[int, float] = {}
    first_seen: dict[int, Hit] = {}
    for ranking in rankings:
        for rank, hit in enumerate(ranking, start=1):
            scores[hit.chunk_id] = scores.get(hit.chunk_id, 0.0) + 1.0 / (k + rank)
            first_seen.setdefault(hit.chunk_id, hit)
    ordered = sorted(scores, key=lambda cid: (-scores[cid], cid))[:limit]
    return tuple(replace(first_seen[cid], score=round(scores[cid], 6)) for cid in ordered)


def within_budget(hits: Sequence[Hit], *, max_chars: int) -> tuple[Hit, ...]:
    """Keep hits in order until the next one would exceed the prompt budget (always keep one)."""
    kept: list[Hit] = []
    used = 0
    for hit in hits:
        if kept and used + len(hit.text) > max_chars:
            break
        kept.append(hit)
        used += len(hit.text)
    return tuple(kept)


def recall_at_k(results: Sequence[Sequence[str]], expected: Sequence[str], *, k: int) -> float:
    """Share of questions whose expected passage key appears in the top k results."""
    if not expected:
        return 0.0
    found = sum(1 for got, want in zip(results, expected, strict=True) if want in got[:k])
    return found / len(expected)
