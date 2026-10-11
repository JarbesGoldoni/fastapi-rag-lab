"""The composition root: the one place that picks adapters and binds them into pipelines."""

from __future__ import annotations

import asyncio
import functools
from collections.abc import Sequence
from dataclasses import dataclass

from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from raglab.adapters.embeddings import local_embedder, openai_embedder
from raglab.adapters.pdf import pymupdf_extractor
from raglab.adapters.registry import Providers, discover
from raglab.db.repository import make_save_chunks, make_text_search, make_vector_search
from raglab.domain import Hit, IngestResult, RawDocument
from raglab.pipelines.ask import hybrid_search
from raglab.pipelines.concurrency import ignore_progress
from raglab.pipelines.ingest import ingest
from raglab.ports import Embedder, IngestOptions, Report
from raglab.settings import Settings


@dataclass(frozen=True, slots=True)
class Container:
    settings: Settings
    engine: AsyncEngine
    embedder: Embedder
    providers: Providers

    async def ingest(
        self,
        doc: RawDocument,
        options: IngestOptions,
        report: Report = ignore_progress,
        job_id: int | None = None,
    ) -> IngestResult:
        return await ingest(
            doc,
            options,
            extractor=pymupdf_extractor,
            embedder=self.embedder,
            save=make_save_chunks(self.engine, job_id),
            report=report,
        )

    async def search(
        self, query: str, *, k: int = 6, document_ids: Sequence[str] | None = None
    ) -> tuple[Hit, ...]:
        return await hybrid_search(
            query,
            embedder=self.embedder,
            vector_search=make_vector_search(self.engine, self.embedder.model),
            text_search=make_text_search(self.engine),
            k=k,
            document_ids=document_ids,
        )

    def ingest_options(self, raw: dict[str, object]) -> IngestOptions:
        """Turn stored job options (provider ids) into live providers; unknown ids are ignored."""
        generators, transcribers = self.providers.generators, self.providers.transcribers
        context_id, extract_id = raw.get("contextualize_with"), raw.get("ai_extraction")
        return IngestOptions(
            contextualize_with=generators.get(context_id) if isinstance(context_id, str) else None,
            ai_extractor=transcribers.get(extract_id) if isinstance(extract_id, str) else None,
        )


def make_embedder(settings: Settings) -> Embedder:
    if settings.embedding_provider == "openai":
        if not settings.openai_api_key:
            raise RuntimeError("EMBEDDING_PROVIDER=openai needs OPENAI_API_KEY")
        return openai_embedder(
            settings.openai_api_key.get_secret_value(),
            settings.openai_embedding_model,
            settings.embedding_dimensions,
        )
    return local_embedder(settings.local_embedding_model)


async def build(settings: Settings, *, embedder: Embedder | None = None) -> Container:
    engine = create_async_engine(settings.database_url, pool_size=10, max_overflow=5)
    chosen = embedder or await asyncio.to_thread(functools.partial(make_embedder, settings))
    if chosen.dimensions != settings.embedding_dimensions:
        raise RuntimeError(
            f"{chosen.model} produces {chosen.dimensions}-dim vectors; "
            f"the schema expects {settings.embedding_dimensions}"
        )
    return Container(
        settings=settings, engine=engine, embedder=chosen, providers=discover(settings)
    )
