"""Embedding adapters. Local (fastembed, ONNX on CPU) needs no key; OpenAI is optional."""

from __future__ import annotations

import asyncio
from collections.abc import Sequence

from raglab.domain import Vector
from raglab.ports import Embedder


def local_embedder(model_name: str, *, cache_dir: str | None = None) -> Embedder:
    """Load the model once (downloads ~70 MB on first run) and expose it as an Embedder."""
    from fastembed import TextEmbedding

    model = TextEmbedding(model_name=model_name, cache_dir=cache_dir)
    dimensions = len(next(iter(model.embed(["probe"]))))

    def _passages(texts: Sequence[str]) -> list[Vector]:
        return [tuple(float(x) for x in v) for v in model.passage_embed(list(texts))]

    def _queries(texts: Sequence[str]) -> list[Vector]:
        return [tuple(float(x) for x in v) for v in model.query_embed(list(texts))]

    async def documents(texts: Sequence[str]) -> list[Vector]:
        return await asyncio.to_thread(_passages, texts)

    async def query(texts: Sequence[str]) -> list[Vector]:
        return await asyncio.to_thread(_queries, texts)

    return Embedder(model=model_name, dimensions=dimensions, documents=documents, query=query)


def openai_embedder(api_key: str, model_name: str, dimensions: int) -> Embedder:
    from openai import AsyncOpenAI

    client = AsyncOpenAI(api_key=api_key)

    async def embed(texts: Sequence[str]) -> list[Vector]:
        response = await client.embeddings.create(
            model=model_name, input=list(texts), dimensions=dimensions
        )
        return [tuple(item.embedding) for item in sorted(response.data, key=lambda d: d.index)]

    return Embedder(
        model=f"openai:{model_name}:{dimensions}",
        dimensions=dimensions,
        documents=embed,
        query=embed,
    )
