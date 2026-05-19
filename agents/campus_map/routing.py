from __future__ import annotations

from dataclasses import dataclass

from agents.campus_map.repository import CampusMapRepository, get_campus_map_repository


_LOCATION_TERMS = (
    "어디",
    "위치",
    "가는길",
    "가는 길",
    "길찾기",
    "길 안내",
    "출입구",
    "정문",
    "후문",
    "몇층",
    "몇 층",
    "찾아가",
)
_BOOK_LOCATION_TERMS = ("책", "도서", "청구기호", "서가", "소장", "자료실")


@dataclass(frozen=True)
class CampusMapEvidence:
    score: float
    reason: str


def collect_campus_map_evidence(
    message: str,
    *,
    repository: CampusMapRepository | None = None,
) -> CampusMapEvidence:
    text = (message or "").strip()
    compact = "".join(text.split())
    if not compact:
        return CampusMapEvidence(score=0.0, reason="empty campus map query")
    if any(term in compact for term in _BOOK_LOCATION_TERMS) and not any(term in compact for term in ("도서관", "학술정보관")):
        return CampusMapEvidence(score=0.0, reason="book shelf/location query belongs to library agent")

    repo = repository or get_campus_map_repository()
    keyword = _rough_place_keyword(text)
    matches = repo.search_places(keyword, limit=3)
    has_location_term = any(term.replace(" ", "") in compact for term in _LOCATION_TERMS)
    if not matches:
        score = 0.35 if has_location_term else 0.0
        return CampusMapEvidence(score=score, reason=f"campus location terms={has_location_term} no place match keyword={keyword}")

    top = matches[0]
    score = top.score
    if has_location_term:
        score = max(score, 0.72)
    elif top.score >= 0.9:
        score = max(score, 0.62)
    reason = (
        f"campus place match={top.place.get('name')} matched={top.matched_text} "
        f"place_score={top.score:.3f} location_terms={has_location_term}"
    )
    return CampusMapEvidence(score=round(min(1.0, score), 3), reason=reason)


def _rough_place_keyword(message: str) -> str:
    text = message
    for token in (
        "내 위치에서",
        "현재 위치에서",
        "어디야",
        "어디 있어",
        "어디있어",
        "위치",
        "가는 길",
        "가는길",
        "길찾기",
        "알려줘",
        "가려면",
        "까지",
        "출입구",
        "정문",
        "후문",
        "?",
    ):
        text = text.replace(token, " ")
    if "에서" in text:
        text = text.split("에서")[-1]
    return " ".join(text.split()) or message

