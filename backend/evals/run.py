"""Retrieval eval: recall@k for vector-only, keyword-only and hybrid search.

Ingests samples/northwind_handbook.pdf into the configured database (idempotent), then asks
every question in questions.jsonl and checks whether the expected page is in the top k.

    uv run python evals/run.py [--k 3]
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from raglab.core.retrieval import fuse_rrf, recall_at_k
from raglab.core.text import document_id, or_tsquery
from raglab.db import repository as repo
from raglab.domain import Hit, RawDocument
from raglab.settings import Settings
from raglab.wiring import Container, build
from raglab.worker import process

HERE = Path(__file__).parent
SAMPLE = HERE.parent / "samples" / "northwind_handbook.pdf"


async def ensure_ingested(container: Container) -> str:
    content = SAMPLE.read_bytes()
    doc = RawDocument(id=document_id(content), filename=SAMPLE.name, content=content)
    row, _ = await repo.create_document(container.engine, doc, {})
    while row.status != "ready":
        job = await repo.claim_job(container.engine)
        if job is None:
            if row.status == "failed":
                raise SystemExit(f"ingest failed: {row.error}")
            await asyncio.sleep(0.2)
        else:
            await process(container, job)
        row = await repo.get_document(container.engine, doc.id) or row
    return doc.id


def pages(hits: tuple[Hit, ...] | list[Hit]) -> list[str]:
    return [str(h.page) for h in hits]


async def main(k: int) -> None:
    container = await build(Settings(embedded_worker=False))
    try:
        doc_id = await ensure_ingested(container)
        scope = [doc_id]
        cases = [json.loads(line) for line in (HERE / "questions.jsonl").read_text().splitlines()]
        vector_search = repo.make_vector_search(container.engine, container.embedder.model)
        text_search = repo.make_text_search(container.engine)

        by_vector, by_text, hybrid = [], [], []
        for case in cases:
            (qvec,) = await container.embedder.query([case["question"]])
            v = await vector_search(qvec, 20, scope)
            tsq = or_tsquery(case["question"])
            t = await text_search(tsq, 20, scope) if tsq else []
            by_vector.append(pages(v))
            by_text.append(pages(t))
            hybrid.append(pages(fuse_rrf([v, t], limit=k)))

        expected = [str(c["page"]) for c in cases]
        print(f"{len(cases)} questions, embedding model {container.embedder.model}\n")
        print(f"{'search':<10} recall@1  recall@{k}")
        for name, results in (("vector", by_vector), ("keyword", by_text), ("hybrid", hybrid)):
            r1 = recall_at_k(results, expected, k=1)
            rk = recall_at_k(results, expected, k=k)
            print(f"{name:<10} {r1:>8.0%}  {rk:>8.0%}")

        misses = [
            c["question"]
            for c, got in zip(cases, hybrid, strict=True)
            if str(c["page"]) not in got[:k]
        ]
        if misses:
            print("\nhybrid misses:", *misses, sep="\n  - ")
    finally:
        await container.engine.dispose()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=3)
    asyncio.run(main(parser.parse_args().k))
