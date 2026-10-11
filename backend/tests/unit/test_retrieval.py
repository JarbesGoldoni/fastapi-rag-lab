from raglab.core.retrieval import fuse_rrf, recall_at_k, within_budget
from raglab.domain import Hit


def hit(chunk_id: int, text: str = "x") -> Hit:
    return Hit(chunk_id, "doc", "file.pdf", 1, chunk_id, text, 0.0)


def test_rrf_rewards_chunks_found_by_both_rankings() -> None:
    vector = [hit(1), hit(2), hit(3)]
    words = [hit(3), hit(4)]
    fused = fuse_rrf([vector, words], k=60, limit=10)
    assert [h.chunk_id for h in fused] == [3, 1, 2, 4]
    assert fused[0].score > fused[1].score


def test_rrf_breaks_ties_by_chunk_id_and_limits() -> None:
    fused = fuse_rrf([[hit(9)], [hit(4)]], limit=1)
    assert [h.chunk_id for h in fused] == [4]


def test_within_budget_keeps_order_and_at_least_one() -> None:
    hits = [hit(1, "a" * 50), hit(2, "b" * 50), hit(3, "c" * 50)]
    assert [h.chunk_id for h in within_budget(hits, max_chars=110)] == [1, 2]
    assert [h.chunk_id for h in within_budget(hits, max_chars=10)] == [1]
    assert within_budget([], max_chars=10) == ()


def test_recall_at_k() -> None:
    results = [["a", "b"], ["c", "d"], ["e"]]
    assert recall_at_k(results, ["b", "x", "e"], k=2) == 2 / 3
    assert recall_at_k(results, ["b", "x", "e"], k=1) == 1 / 3
