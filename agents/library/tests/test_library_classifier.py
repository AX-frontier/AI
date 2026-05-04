from __future__ import annotations

from agents.library.classifier import (
    RetrievalEvidence,
    classify_intent,
    extract_search_keyword,
    load_intent_config,
    score_intents,
)


def test_loads_yaml_intent_dictionary() -> None:
    config = load_intent_config()

    intents = {rule.intent for rule in config.rules}
    assert {
        "BOOK_SEARCH",
        "BOOK_LOCATION",
        "LIBRARY_GUIDE",
        "BOOK_RECOMMENDATION",
        "LIBRARY_GENERAL",
    }.issubset(intents)
    assert "도서 검색" in config.keyword_cleanup


def test_guide_question_is_not_forced_to_book_search() -> None:
    result = classify_intent(
        "도서관 운영 시간 알려줘",
        evidence=RetrievalEvidence(guide_hits=1),
    )

    assert result.intent == "LIBRARY_GUIDE"
    assert result.confidence >= 0.75


def test_book_search_and_location_are_classified_separately() -> None:
    search = classify_intent(
        "파이썬 책 찾아줘",
        evidence=RetrievalEvidence(book_hits=1),
    )
    location = classify_intent(
        "파이썬 자료구조 어디 있어?",
        evidence=RetrievalEvidence(book_hits=1),
    )

    assert search.intent == "BOOK_SEARCH"
    assert location.intent == "BOOK_LOCATION"


def test_retrieval_evidence_boosts_matching_intent_confidence() -> None:
    without_evidence = score_intents("파이썬", RetrievalEvidence())
    with_evidence = score_intents("파이썬", RetrievalEvidence(book_hits=1))

    search_without = next(score for score in without_evidence if score.intent == "BOOK_SEARCH")
    search_with = next(score for score in with_evidence if score.intent == "BOOK_SEARCH")

    assert search_with.confidence > search_without.confidence


def test_ambiguous_question_deterministically_falls_back_without_llm() -> None:
    result = classify_intent("궁금한 게 있어", evidence=RetrievalEvidence())

    assert result.intent == "LIBRARY_GENERAL"
    assert result.used_llm is False
    assert result.ambiguous is True


def test_extract_search_keyword_uses_yaml_cleanup_phrases() -> None:
    assert extract_search_keyword("파이썬 도서 검색") == "파이썬"

