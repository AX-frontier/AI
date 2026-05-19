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
    assert result.ambiguous is False


def test_library_plus_book_question_prefers_book_search() -> None:
    result = classify_intent(
        "도서관에 파이썬 책 있어?",
        evidence=RetrievalEvidence(book_hits=1, guide_hits=1),
    )

    assert result.intent == "BOOK_SEARCH"


def test_general_help_message_prefers_library_general_even_with_guide_hits() -> None:
    result = classify_intent(
        "도와줘",
        evidence=RetrievalEvidence(guide_hits=1),
    )

    assert result.intent == "LIBRARY_GENERAL"


def test_shelf_question_prefers_book_location() -> None:
    result = classify_intent(
        "1-A-1-d 서가 책 찾아줘",
        evidence=RetrievalEvidence(book_hits=1),
    )

    assert result.intent == "BOOK_LOCATION"


def test_document_review_phrase_does_not_match_shelf_keyword() -> None:
    scores = score_intents("문서가 있는데 검토해줘", RetrievalEvidence(guide_hits=1))
    location = next(score for score in scores if score.intent == "BOOK_LOCATION")

    assert "서가" not in location.matched_keywords
    assert location.confidence < 0.7


def test_book_shelf_question_still_prefers_book_location() -> None:
    result = classify_intent(
        "파이썬 책 서가 어디야?",
        evidence=RetrievalEvidence(book_hits=1),
    )

    assert result.intent == "BOOK_LOCATION"


def test_library_shelf_location_question_still_prefers_book_location() -> None:
    result = classify_intent(
        "자료실 서가 위치 알려줘",
        evidence=RetrievalEvidence(book_hits=1),
    )

    assert result.intent == "BOOK_LOCATION"


def test_library_book_location_question_prefers_book_location() -> None:
    result = classify_intent(
        "도서관에 파이썬 책 어디 있어?",
        evidence=RetrievalEvidence(book_hits=1, guide_hits=1),
    )

    assert result.intent == "BOOK_LOCATION"


def test_overdue_fee_where_question_prefers_library_guide() -> None:
    result = classify_intent(
        "연체료는 어디서 채납해?",
        evidence=RetrievalEvidence(guide_hits=1),
    )

    assert result.intent == "LIBRARY_GUIDE"


def test_library_policy_phrase_prefers_library_guide_even_with_book_hits() -> None:
    result = classify_intent(
        "학술정보관규정 알려줘",
        evidence=RetrievalEvidence(book_hits=5, guide_hits=1),
    )

    assert result.intent == "LIBRARY_GUIDE"


def test_staff_responsibility_question_prefers_library_guide_even_with_book_hits() -> None:
    result = classify_intent(
        "목록, 장서관리정책 수립담당자는 누구야?",
        evidence=RetrievalEvidence(book_hits=5, guide_hits=1),
    )

    assert result.intent == "LIBRARY_GUIDE"


def test_recommendation_question_prefers_book_recommendation() -> None:
    result = classify_intent(
        "도서관에서 볼만한 파이썬 책 추천해줘",
        evidence=RetrievalEvidence(book_hits=1, guide_hits=1),
    )

    assert result.intent == "BOOK_RECOMMENDATION"


def test_popular_book_question_prefers_book_recommendation_without_results() -> None:
    result = classify_intent(
        "가장 인기있는 책은 뭐야?",
        evidence=RetrievalEvidence(),
    )

    assert result.intent == "BOOK_RECOMMENDATION"


def test_physical_book_search_beats_ebook_guide_term() -> None:
    result = classify_intent(
        "전자책 말고 종이책 찾아줘",
        evidence=RetrievalEvidence(guide_hits=1),
    )

    assert result.intent == "BOOK_SEARCH"


def test_general_inquiry_phrase_prefers_library_general() -> None:
    result = classify_intent(
        "문의하고 싶은 게 있어",
        evidence=RetrievalEvidence(guide_hits=1),
    )

    assert result.intent == "LIBRARY_GENERAL"


def test_recommendation_negation_prefers_book_search() -> None:
    result = classify_intent(
        "추천 말고 파이썬 책만 찾아줘",
        evidence=RetrievalEvidence(book_hits=1),
    )

    assert result.intent == "BOOK_SEARCH"


def test_book_topic_word_prefers_book_search_without_command_verb() -> None:
    result = classify_intent(
        "파이썬 말고 자바 입문서",
        evidence=RetrievalEvidence(book_hits=1),
    )

    assert result.intent == "BOOK_SEARCH"


def test_extract_search_keyword_uses_yaml_cleanup_phrases() -> None:
    assert extract_search_keyword("파이썬 도서 검색") == "파이썬"


def test_extract_search_keyword_strips_trailing_book_indicator_for_book_intents() -> None:
    assert extract_search_keyword("클린 코드 책 어디 있어?", "BOOK_LOCATION") == "클린 코드"
    assert extract_search_keyword("컴퓨터 과학 도서를 찾아줘", "BOOK_SEARCH") == "컴퓨터 과학"
    assert extract_search_keyword("파이썬 도서 추천해줘", "BOOK_RECOMMENDATION") == "파이썬"
    assert extract_search_keyword("도서관에 파이썬 책 있어?", "BOOK_SEARCH") == "파이썬"


def test_extract_search_keyword_strips_trailing_book_field_qualifiers() -> None:
    assert extract_search_keyword("슈카친구들 저자 도서 검색", "BOOK_SEARCH") == "슈카친구들"
    assert extract_search_keyword("Facet 출판사 도서 검색", "BOOK_SEARCH") == "Facet"
    assert extract_search_keyword("001 ㄱ785ㄷ 청구기호 위치 알려줘", "BOOK_LOCATION") == "001 ㄱ785ㄷ"
    assert extract_search_keyword("1-A-1-d 서가 위치 알려줘", "BOOK_LOCATION") == "1-A-1-d"


def test_extract_search_keyword_keeps_generic_book_term_for_short_book_queries() -> None:
    assert extract_search_keyword("책 있어?", "BOOK_SEARCH") == "책"
    assert extract_search_keyword("책 찾아줘", "BOOK_SEARCH") == "책"
    assert extract_search_keyword("학술정보관 책", "BOOK_SEARCH") == "책"
    assert extract_search_keyword("학술 정보관 책", "BOOK_SEARCH") == "책"


def test_extract_search_keyword_prefers_replacement_topic_after_exceptive_phrase() -> None:
    assert extract_search_keyword("파이썬 말고 자바 입문서 찾아줘", "BOOK_SEARCH") == "자바 입문"


def test_extract_search_keyword_without_intent_preserves_trailing_book_indicator() -> None:
    assert extract_search_keyword("클린 코드 책 어디 있어?") == "클린 코드 책"
