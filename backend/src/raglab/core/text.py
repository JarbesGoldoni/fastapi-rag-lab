"""Pure text helpers: identity, cleanup and query building."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Sequence

from raglab.domain import Page

_HYPHEN_BREAK = re.compile(r"(\w)-\n(\w)")
_SPACES = re.compile(r"[ \t\f\v\u00a0]+")
_MANY_NEWLINES = re.compile(r"\n{3,}")
_WORD = re.compile(r"[^\W_]+", re.UNICODE)

# Short English + Portuguese stopword list. Enough to keep full-text search from
# matching every chunk on "what", "the", "como", "que".
STOPWORDS = frozenset(
    """
    a an and are as at be but by can could did do does for from had has have how i if in into
    is it its me my no not of on or our so than that the their them then there these they this
    to was we were what when where which who why will with would you your about tell explain
    o os um uma uns umas de do da dos das e em na nos nas por para com sem que qual quais
    quem como onde quando porque se ao aos é são foi ser ter tem meu minha seu sua isso isto
    """.split()
)


def document_id(content: bytes) -> str:
    """Content-addressed ID: the same file always gets the same ID, so uploads are idempotent."""
    return hashlib.sha256(content).hexdigest()


def clean_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _HYPHEN_BREAK.sub(r"\1\2", text)
    text = _SPACES.sub(" ", text)
    text = "\n".join(line.strip() for line in text.split("\n"))
    return _MANY_NEWLINES.sub("\n\n", text).strip()


def clean_pages(pages: Sequence[Page]) -> tuple[Page, ...]:
    cleaned = (Page(number=p.number, text=clean_text(p.text)) for p in pages)
    return tuple(p for p in cleaned if p.text)


def keywords(query: str, *, min_length: int = 3) -> tuple[str, ...]:
    """Distinct, lowercased, order-preserving search terms without stopwords."""
    seen: dict[str, None] = {}
    for word in _WORD.findall(query.lower()):
        if len(word) >= min_length and word not in STOPWORDS:
            seen.setdefault(word, None)
    return tuple(seen)


def or_tsquery(query: str) -> str:
    """A to_tsquery() string that matches any keyword. Empty when nothing is searchable.

    websearch_to_tsquery() ANDs every word, which almost never matches a natural question.
    """
    return " | ".join(keywords(query))


def total_chars(pages: Sequence[Page]) -> int:
    return sum(len(p.text) for p in pages)
