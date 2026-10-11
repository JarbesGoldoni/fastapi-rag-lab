import itertools

import pytest
from hypothesis import given
from hypothesis import strategies as st

from raglab.core.chunking import chunk_page, chunk_pages, windows, with_contexts
from raglab.domain import Chunk, ChunkConfig, Page

words_strategy = st.lists(
    st.text(alphabet=st.characters(categories=["L", "N"]), min_size=1, max_size=30), max_size=300
)


@given(words=words_strategy, size=st.integers(40, 400), overlap_ratio=st.floats(0, 0.5))
def test_windows_cover_every_word_in_order(
    words: list[str], size: int, overlap_ratio: float
) -> None:
    overlap = int(size * overlap_ratio)
    ranges = list(windows(words, size=size, overlap=overlap))
    covered = [i for start, end in ranges for i in range(start, end)]
    assert sorted(set(covered)) == list(range(len(words)))
    assert all(a[0] < b[0] for a, b in itertools.pairwise(ranges))


@given(words=words_strategy, size=st.integers(40, 400))
def test_windows_respect_size_unless_single_long_word(words: list[str], size: int) -> None:
    for start, end in windows(words, size=size, overlap=size // 4):
        text = " ".join(words[start:end])
        assert len(text) <= size or end - start == 1


def test_consecutive_chunks_share_the_overlap() -> None:
    page = Page(1, " ".join(f"w{i:03d}" for i in range(100)))  # 4-char words
    first, second, *_ = chunk_page(page, cfg=ChunkConfig(size=100, overlap=20))
    assert first.split()[-4:] == second.split()[:4]


def test_chunks_never_cross_pages_and_ordinals_are_global() -> None:
    pages = [Page(1, "alpha " * 50), Page(3, "beta " * 50)]
    chunks = chunk_pages("doc", pages, cfg=ChunkConfig(size=120, overlap=0))
    assert [c.ordinal for c in chunks] == list(range(len(chunks)))
    assert {c.page for c in chunks if "alpha" in c.text} == {1}
    assert {c.page for c in chunks if "beta" in c.text} == {3}


def test_chunking_is_deterministic() -> None:
    pages = [Page(1, "some text " * 200)]
    cfg = ChunkConfig(size=200, overlap=40)
    assert chunk_pages("d", pages, cfg=cfg) == chunk_pages("d", pages, cfg=cfg)


def test_invalid_config_is_rejected() -> None:
    with pytest.raises(ValueError):
        list(windows(["a"], size=10, overlap=10))


def test_with_contexts_attaches_by_position_and_skips_blanks() -> None:
    chunks = (Chunk("d", 0, 1, "a"), Chunk("d", 1, 1, "b"), Chunk("d", 2, 1, "c"))
    result = with_contexts(chunks, ["ctx a", "  "])
    assert [c.context for c in result] == ["ctx a", "", ""]
    assert result[0].embedding_text == "ctx a\n\na"
    assert result[1].embedding_text == "b"
