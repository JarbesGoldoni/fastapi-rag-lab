"""The composition root: picks adapters once and binds them into the application's use cases.

Routes and the worker talk to the Container only; nothing outside db/ and wiring touches
the engine.
"""

from __future__ import annotations

import asyncio
import functools
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass
from typing import Any

from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from raglab.adapters.embeddings import local_embedder, openai_embedder
from raglab.adapters.pdf import pymupdf_extractor
from raglab.adapters.registry import Providers, discover
from raglab.core.prompts import answer_prompt, parse_rewrite
from raglab.core.retrieval import within_budget
from raglab.db import repository as repo
from raglab.db.repository import DocumentRow, Job
from raglab.db.schema import EMBEDDING_DIMENSIONS
from raglab.domain import Hit, IngestResult, Prompt, RawDocument, Turn
from raglab.pipelines.ask import AskEvent, answer, hybrid_search
from raglab.pipelines.concurrency import ignore_progress
from raglab.pipelines.ingest import ingest
from raglab.ports import Embedder, Generator, IngestOptions, Report
from raglab.settings import Settings

BROWSER_CONTEXT_CHARS = 4_000  # Gemini Nano has a small context window


@dataclass(frozen=True, slots=True)
class Container:
    settings: Settings
    engine: AsyncEngine
    embedder: Embedder
    providers: Providers

    # ------------------------------------------------------------ documents

    async def add_document(
        self, doc: RawDocument, options: dict[str, Any]
    ) -> tuple[DocumentRow, bool]:
        return await repo.create_document(self.engine, doc, options)

    async def list_documents(self) -> list[DocumentRow]:
        return await repo.list_documents(self.engine)

    async def get_document(self, document_id: str) -> DocumentRow | None:
        return await repo.get_document(self.engine, document_id)

    async def get_file(self, document_id: str) -> RawDocument | None:
        return await repo.get_raw_document(self.engine, document_id)

    async def reprocess(self, document_id: str, options: dict[str, Any]) -> DocumentRow | None:
        if await repo.get_document(self.engine, document_id) is None:
            return None
        await repo.requeue(self.engine, document_id, options)
        return await repo.get_document(self.engine, document_id)

    async def delete_document(self, document_id: str) -> bool:
        return await repo.delete_document(self.engine, document_id)

    async def ingest(
        self,
        doc: RawDocument,
        options: IngestOptions,
        report: Report = ignore_progress,
        job: Job | None = None,
    ) -> IngestResult:
        return await ingest(
            doc,
            options,
            extractor=pymupdf_extractor,
            embedder=self.embedder,
            save=repo.make_save_chunks(self.engine, job),
            report=report,
        )

    def ingest_options(self, raw: dict[str, object]) -> IngestOptions:
        """Turn stored job options (provider ids) into live providers; unknown ids are ignored."""
        generators, transcribers = self.providers.generators, self.providers.transcribers
        context_id, extract_id = raw.get("contextualize_with"), raw.get("ai_extraction")
        return IngestOptions(
            contextualize_with=generators.get(context_id) if isinstance(context_id, str) else None,
            ai_extractor=transcribers.get(extract_id) if isinstance(extract_id, str) else None,
        )

    # ------------------------------------------------------------ questions

    async def search(
        self, query: str, *, k: int = 6, document_ids: Sequence[str] | None = None
    ) -> tuple[Hit, ...]:
        return await hybrid_search(
            query,
            embedder=self.embedder,
            vector_search=repo.make_vector_search(self.engine, self.embedder.model),
            text_search=repo.make_text_search(self.engine),
            k=k,
            document_ids=document_ids,
        )

    def answer(
        self,
        question: str,
        *,
        generator: Generator,
        history: Sequence[Turn] = (),
        k: int = 6,
        document_ids: Sequence[str] | None = None,
    ) -> AsyncIterator[AskEvent]:
        return answer(
            question,
            search=functools.partial(self.search, k=k, document_ids=document_ids),
            generator=generator,
            history=history,
        )

    async def browser_context(
        self,
        question: str,
        *,
        rewritten: str | None = None,
        history: Sequence[Turn] = (),
        k: int = 6,
        document_ids: Sequence[str] | None = None,
    ) -> tuple[tuple[Hit, ...], Prompt]:
        """Sources and prompt for a model that runs in the browser. The browser rewrites
        follow-ups itself and sends the raw rewrite."""
        query = parse_rewrite(rewritten, fallback=question) if rewritten else question
        hits = within_budget(
            await self.search(query, k=k, document_ids=document_ids),
            max_chars=BROWSER_CONTEXT_CHARS,
        )
        return hits, answer_prompt(question, hits, history)


def make_embedder(settings: Settings) -> Embedder:
    if settings.embedding_provider == "openai":
        if not settings.openai_api_key:
            raise RuntimeError("EMBEDDING_PROVIDER=openai needs OPENAI_API_KEY")
        return openai_embedder(
            settings.openai_api_key.get_secret_value(),
            settings.openai_embedding_model,
            EMBEDDING_DIMENSIONS,
        )
    return local_embedder(settings.local_embedding_model)


async def build(settings: Settings, *, embedder: Embedder | None = None) -> Container:
    engine = create_async_engine(settings.database_url, pool_size=10, max_overflow=5)
    chosen = embedder or await asyncio.to_thread(functools.partial(make_embedder, settings))
    if chosen.dimensions != EMBEDDING_DIMENSIONS:
        raise RuntimeError(
            f"{chosen.model} produces {chosen.dimensions}-dim vectors; "
            f"the database schema stores {EMBEDDING_DIMENSIONS}"
        )
    return Container(
        settings=settings, engine=engine, embedder=chosen, providers=discover(settings)
    )
