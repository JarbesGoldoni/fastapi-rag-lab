from raglab.core.text import clean_pages, clean_text, document_id, keywords, or_tsquery
from raglab.domain import Page


def test_document_id_is_content_addressed() -> None:
    assert document_id(b"same") == document_id(b"same")
    assert document_id(b"same") != document_id(b"other")
    assert len(document_id(b"x")) == 64


def test_clean_text_joins_hyphenated_breaks_and_collapses_space() -> None:
    raw = "Pay-\nment   orches-\ntration\r\n\n\n\n\nnext\u00a0 line  "
    assert clean_text(raw) == "Payment orchestration\n\nnext line"


def test_clean_pages_drops_empty_pages_and_keeps_numbers() -> None:
    pages = clean_pages([Page(1, "  "), Page(2, "text "), Page(3, "\n\n")])
    assert pages == (Page(2, "text"),)


def test_keywords_drop_stopwords_short_words_and_duplicates() -> None:
    assert keywords("What is the refund policy? The REFUND window, ok?") == (
        "refund",
        "policy",
        "window",
    )
    assert keywords("Qual é a política de reembolso?") == ("política", "reembolso")


def test_or_tsquery_matches_any_keyword_and_is_safe() -> None:
    assert or_tsquery("refund policy & window!") == "refund | policy | window"
    assert or_tsquery("what is it?") == ""
    assert or_tsquery("'; drop table chunks; --") == "drop | table | chunks"
