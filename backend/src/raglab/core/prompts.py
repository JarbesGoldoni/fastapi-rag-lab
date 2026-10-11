"""Prompt builders and parsers for model output. All pure."""

from __future__ import annotations

import json
import re
from collections.abc import Sequence

from raglab.domain import Chunk, Citation, Hit, Page, Prompt, Turn

ANSWER_SYSTEM = """You answer questions about the user's documents.
Use only the numbered sources provided. Cite every claim with its source number in square
brackets, like [1] or [2][3]. If the sources do not contain the answer, say so plainly.
Answer in the language of the question. Be concise: short paragraphs, no headings.
Your answer may be read aloud, so avoid tables and long lists."""

CONTEXT_SYSTEM = """You help a search engine index document passages.
For each numbered passage, write one sentence that situates it within the whole document
(what section or topic it belongs to, and what it is about). Return only a JSON array of
strings, one per passage, in the same order. No commentary."""

TRANSCRIBE_SYSTEM = """You transcribe PDF documents. Return only a JSON array of objects
{"page": <1-based page number>, "text": "<all readable text on that page>"}. Keep the
original language. No commentary."""

_CITATION = re.compile(r"\[(\d{1,3})\]")


def format_sources(hits: Sequence[Hit]) -> str:
    return "\n\n".join(
        f"[{i}] {hit.filename}, page {hit.page}\n{hit.text}" for i, hit in enumerate(hits, start=1)
    )


def without_citations(text: str) -> str:
    """Earlier answers cite an earlier source list; their [n] markers would mislead now."""
    return re.sub(r"\s*\[\d{1,3}\]", "", text)


def format_history(history: Sequence[Turn], *, max_turns: int = 6) -> str:
    recent = history[-max_turns:]
    return "\n".join(f"{t.role.capitalize()}: {without_citations(t.text)}" for t in recent)


def answer_prompt(question: str, hits: Sequence[Hit], history: Sequence[Turn] = ()) -> Prompt:
    parts = [f"Sources:\n\n{format_sources(hits)}"]
    if history:
        parts.append(f"Conversation so far:\n{format_history(history)}")
    parts.append(f"Question: {question}")
    return Prompt(system=ANSWER_SYSTEM, user="\n\n---\n\n".join(parts))


def context_prompt(document_excerpt: str, chunks: Sequence[Chunk], *, preview: int = 400) -> Prompt:
    passages = "\n\n".join(f"{i}. {c.text[:preview]}" for i, c in enumerate(chunks, start=1))
    user = f"Document:\n{document_excerpt}\n\n---\n\nPassages:\n\n{passages}"
    return Prompt(system=CONTEXT_SYSTEM, user=user)


def transcribe_prompt(filename: str) -> Prompt:
    return Prompt(
        system=TRANSCRIBE_SYSTEM,
        user=f"Transcribe every page of the PDF file `{filename}` in the current directory.",
    )


def extract_json(text: str) -> object | None:
    """Parse the first JSON array or object in model output, tolerating code fences and prose."""
    for opener, closer in (("[", "]"), ("{", "}")):
        start, end = text.find(opener), text.rfind(closer)
        if start != -1 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                continue
    return None


def parse_contexts(text: str, *, expected: int) -> tuple[str, ...]:
    data = extract_json(text)
    if not isinstance(data, list):
        return ()
    return tuple(str(item) if isinstance(item, str) else "" for item in data[:expected])


def parse_transcription(text: str) -> tuple[Page, ...]:
    data = extract_json(text)
    if not isinstance(data, list):
        return ()
    pages = (
        Page(number=int(item["page"]), text=str(item["text"]))
        for item in data
        if isinstance(item, dict) and "page" in item and "text" in item
    )
    return tuple(sorted(pages, key=lambda p: p.number))


def citations(answer: str, hits: Sequence[Hit]) -> tuple[Citation, ...]:
    """The sources the answer actually cited, in first-mention order, ignoring invalid numbers."""
    seen: dict[int, None] = {}
    for match in _CITATION.finditer(answer):
        n = int(match.group(1))
        if 1 <= n <= len(hits):
            seen.setdefault(n, None)
    return tuple(Citation(number=n, hit=hits[n - 1]) for n in seen)
