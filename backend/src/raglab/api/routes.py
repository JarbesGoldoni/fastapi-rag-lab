"""HTTP routes. Thin: validate, call a pipeline or repository function, convert."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Annotated

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, Response, UploadFile
from fastapi.responses import StreamingResponse

from raglab.api.schemas import (
    AskIn,
    ContextIn,
    ContextOut,
    DocumentOut,
    HitOut,
    PromptOut,
    ProviderOut,
    ProvidersOut,
    ReprocessIn,
    RewriteIn,
    SearchIn,
    citation_out,
    context_out,
    document_out,
    hit_out,
    turns,
)
from raglab.core.events import sse
from raglab.core.prompts import rewrite_prompt
from raglab.core.text import document_id
from raglab.domain import RawDocument
from raglab.pipelines.ask import Delta, Done, Reset, Sources
from raglab.wiring import Container

router = APIRouter(prefix="/api")


def container(request: Request) -> Container:
    return request.app.state.container


Dep = Annotated[Container, Depends(container)]


def _job_options(
    c: Container, contextualize_with: str | None, ai_extraction: str | None
) -> dict[str, str]:
    """Validate provider ids from the form and keep only the ones that exist."""
    options: dict[str, str] = {}
    if contextualize_with:
        if contextualize_with not in c.providers.generators:
            raise HTTPException(422, f"Unknown provider: {contextualize_with}")
        options["contextualize_with"] = contextualize_with
    if ai_extraction:
        if ai_extraction not in c.providers.transcribers:
            raise HTTPException(422, f"No AI extraction with: {ai_extraction}")
        options["ai_extraction"] = ai_extraction
    return options


@router.get("/health")
async def health(request: Request) -> dict[str, str]:
    task = request.app.state.worker
    worker = "external" if task is None else "stopped" if task.done() else "running"
    return {"status": "ok" if worker != "stopped" else "degraded", "worker": worker}


@router.get("/providers")
async def providers(c: Dep) -> ProvidersOut:
    return ProvidersOut(
        generators=[
            ProviderOut(id=g.id, label=g.label, kind=g.kind)
            for g in c.providers.generators.values()
        ],
        transcribers=list(c.providers.transcribers),
        embedding_model=c.embedder.model,
        default_generator=c.providers.default_generator,
    )


@router.get("/documents")
async def list_documents(c: Dep) -> list[DocumentOut]:
    return [document_out(row) for row in await c.list_documents()]


@router.post("/documents", status_code=202)
async def upload(
    c: Dep,
    response: Response,
    file: Annotated[UploadFile, File()],
    contextualize_with: Annotated[str | None, Form()] = None,
    ai_extraction: Annotated[str | None, Form()] = None,
) -> DocumentOut:
    if (file.size or 0) > c.settings.max_upload_mb * 1024 * 1024:
        raise HTTPException(413, f"Files up to {c.settings.max_upload_mb} MB")
    content = await file.read()
    if not content.startswith(b"%PDF"):
        raise HTTPException(415, "Only PDF files are supported")
    doc = RawDocument(
        id=document_id(content), filename=file.filename or "document.pdf", content=content
    )
    row, created = await c.add_document(doc, _job_options(c, contextualize_with, ai_extraction))
    if not created:
        response.status_code = 200
    return document_out(row)


@router.get("/documents/{doc_id}")
async def get_document(c: Dep, doc_id: str) -> DocumentOut:
    row = await c.get_document(doc_id)
    if row is None:
        raise HTTPException(404, "Document not found")
    return document_out(row)


@router.get("/documents/{doc_id}/file")
async def get_file(c: Dep, doc_id: str) -> Response:
    doc = await c.get_file(doc_id)
    if doc is None:
        raise HTTPException(404, "Document not found")
    return Response(
        doc.content,
        media_type="application/pdf",
        headers={"Content-Disposition": f'inline; filename="{doc.filename}"'},
    )


@router.post("/documents/{doc_id}/reprocess", status_code=202)
async def reprocess(c: Dep, doc_id: str, body: ReprocessIn) -> DocumentOut:
    row = await c.reprocess(doc_id, _job_options(c, body.contextualize_with, body.ai_extraction))
    if row is None:
        raise HTTPException(404, "Document not found")
    return document_out(row)


@router.delete("/documents/{doc_id}", status_code=204)
async def delete_document(c: Dep, doc_id: str) -> None:
    if not await c.delete_document(doc_id):
        raise HTTPException(404, "Document not found")


@router.post("/search")
async def search(c: Dep, body: SearchIn) -> list[HitOut]:
    hits = await c.search(body.query, k=body.k, document_ids=body.document_ids)
    return [hit_out(h) for h in hits]


@router.post("/rewrite-prompt")
async def rewrite(body: RewriteIn) -> PromptOut:
    """For browser models: the prompt that turns a follow-up into a standalone query."""
    prompt = rewrite_prompt(body.query, turns(body.history))
    return PromptOut(system=prompt.system, user=prompt.user)


@router.post("/context")
async def context(c: Dep, body: ContextIn) -> ContextOut:
    """Retrieval and prompt only. The browser runs the model (Chrome's built-in Gemini Nano)."""
    hits, prompt = await c.browser_context(
        body.query,
        rewritten=body.rewritten,
        history=turns(body.history),
        k=body.k,
        document_ids=body.document_ids,
    )
    return context_out(hits, prompt)


@router.post("/ask")
async def ask_stream(c: Dep, body: AskIn) -> StreamingResponse:
    generator = c.providers.generators.get(body.provider)
    if generator is None:
        raise HTTPException(422, f"Provider not available: {body.provider}")

    async def events() -> AsyncIterator[bytes]:
        try:
            async for event in c.answer(
                body.query,
                generator=generator,
                history=turns(body.history),
                k=body.k,
                document_ids=body.document_ids,
            ):
                match event:
                    case Sources(hits=selected):
                        yield sse("sources", [hit_out(h).model_dump() for h in selected])
                    case Delta(text=text):
                        yield sse("delta", {"text": text})
                    case Reset():
                        yield sse("reset", {})
                    case Done(answer=answer, citations=cited):
                        yield sse(
                            "done",
                            {
                                "answer": answer,
                                "citations": [citation_out(x).model_dump() for x in cited],
                            },
                        )
        except Exception as exc:  # stream the failure; the HTTP status is already 200
            yield sse("error", {"message": str(exc) or type(exc).__name__})

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
