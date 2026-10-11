"""Text-layer PDF extraction with PyMuPDF. No AI involved."""

from __future__ import annotations

import asyncio

import pymupdf

from raglab.domain import Page, Progress, RawDocument, Stage
from raglab.ports import Extractor, Report

PAGES_PER_STEP = 8


def _open(content: bytes) -> pymupdf.Document:
    return pymupdf.open(stream=content, filetype="pdf")


def _read(pdf: pymupdf.Document, start: int, end: int) -> list[Page]:
    return [
        Page(number=i + 1, text=str(pdf[i].get_text("text", sort=True))) for i in range(start, end)
    ]


async def extract_text_layer(doc: RawDocument, report: Report) -> tuple[Page, ...]:
    """Read a few pages per thread hop, so parsing stays off the event loop and every step
    reports the real number of pages read."""
    pdf = await asyncio.to_thread(_open, doc.content)
    try:
        total = pdf.page_count
        await report(Progress(Stage.READING, 0, total))
        pages: list[Page] = []
        for start in range(0, total, PAGES_PER_STEP):
            end = min(start + PAGES_PER_STEP, total)
            pages.extend(await asyncio.to_thread(_read, pdf, start, end))
            await report(Progress(Stage.READING, end, total))
        return tuple(pages)
    finally:
        pdf.close()


pymupdf_extractor = Extractor(id="pymupdf", extract=extract_text_layer)
