"""FastAPI application factory. Run with: uv run raglab (or uvicorn raglab.main:app)."""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import AsyncGenerator, Awaitable, Callable
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from raglab import worker
from raglab.api.routes import router
from raglab.settings import Settings
from raglab.wiring import Container, build

FRONTEND_DIST = Path(__file__).resolve().parents[3] / "frontend" / "dist"

type BuildContainer = Callable[[Settings], Awaitable[Container]]


def create_app(settings: Settings, *, build_container: BuildContainer = build) -> FastAPI:
    @contextlib.asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncGenerator[None]:
        container = await build_container(settings)
        app.state.container = container
        stop = asyncio.Event()
        task = (
            asyncio.create_task(worker.run(container, stop), name="ingest-worker")
            if settings.embedded_worker
            else None
        )
        try:
            yield
        finally:
            stop.set()
            if task:
                await task
            await container.engine.dispose()

    app = FastAPI(title="RAG Lab", lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(router)
    if FRONTEND_DIST.is_dir():
        app.mount("/", StaticFiles(directory=FRONTEND_DIST, html=True), name="frontend")
    return app


def app() -> FastAPI:
    """Factory for `uvicorn --factory raglab.main:app`."""
    return create_app(Settings())


def run() -> None:
    import uvicorn

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    uvicorn.run("raglab.main:app", factory=True, host="127.0.0.1", port=8000)
