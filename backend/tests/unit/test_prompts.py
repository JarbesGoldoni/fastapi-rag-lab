from raglab.core.prompts import (
    answer_prompt,
    citations,
    extract_json,
    parse_contexts,
    parse_transcription,
)
from raglab.domain import Hit, Page, Turn


def hits() -> list[Hit]:
    return [
        Hit(10, "d", "handbook.pdf", 2, 0, "Refunds take 14 days.", 0.9),
        Hit(11, "d", "handbook.pdf", 5, 1, "Support is open 9-18h.", 0.8),
    ]


def test_answer_prompt_numbers_sources_and_includes_history() -> None:
    prompt = answer_prompt("How long?", hits(), [Turn("user", "hi"), Turn("assistant", "hello")])
    assert "[1] handbook.pdf, page 2\nRefunds take 14 days." in prompt.user
    assert "[2] handbook.pdf, page 5" in prompt.user
    assert "User: hi\nAssistant: hello" in prompt.user
    assert prompt.user.endswith("Question: How long?")


def test_citations_keep_first_mention_order_and_ignore_invalid_numbers() -> None:
    cited = citations("Fourteen days [2]. Also [1][2] and [7].", hits())
    assert [(c.number, c.hit.chunk_id) for c in cited] == [(2, 11), (1, 10)]


def test_extract_json_tolerates_fences_and_prose() -> None:
    assert extract_json('Sure!\n```json\n["a", "b"]\n```') == ["a", "b"]
    assert extract_json("no json here") is None


def test_parse_contexts_truncates_and_blanks_non_strings() -> None:
    assert parse_contexts('["one", 2, "three", "four"]', expected=3) == ("one", "", "three")
    assert parse_contexts("garbage", expected=2) == ()


def test_parse_transcription_sorts_pages_and_skips_bad_items() -> None:
    text = '[{"page": 2, "text": "b"}, {"oops": 1}, {"page": 1, "text": "a"}]'
    assert parse_transcription(text) == (Page(1, "a"), Page(2, "b"))


def test_history_drops_old_citation_markers() -> None:
    prompt = answer_prompt("And instant?", hits(), [Turn("assistant", "Card: 180 days [1][2].")])
    assert "Assistant: Card: 180 days." in prompt.user
