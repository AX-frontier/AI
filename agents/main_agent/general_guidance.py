from __future__ import annotations

from dataclasses import dataclass

from agents.main_agent.api.schemas import MainChatResponse, MainSource


@dataclass(frozen=True)
class GeneralGuideTarget:
    label: str
    url: str
    aliases: tuple[str, ...]
    guide_terms: tuple[str, ...]
    notice_exclusion_terms: tuple[str, ...]
    answer_lines: tuple[str, ...]


GENERAL_GUIDE_TARGETS: tuple[GeneralGuideTarget, ...] = (
    GeneralGuideTarget(
        label="장학금 안내",
        url="https://hansung.ac.kr/edubank/5762/subview.do",
        aliases=("장학금", "장학", "장학금 안내", "장학 안내"),
        guide_terms=(
            "신청",
            "신청방법",
            "신청절차",
            "절차",
            "방법",
            "어떻게",
            "어디서",
            "종류",
            "대상",
            "자격",
            "안내",
        ),
        notice_exclusion_terms=(
            "공지",
            "마감",
            "언제",
            "결과",
            "선발",
            "발표",
            "지급제외",
            "지급 제외",
            "동의서",
            "동의서제출",
            "동의서 제출",
            "국가장학금",
            "국가우수장학금",
            "주거안정장학금",
            "학업장려금",
        ),
        answer_lines=(
            "장학금 신청 방법은 장학 종류와 학기별 공지에 따라 달라집니다.",
            "공통 안내는 장학금 안내 페이지에서 확인하고, 실제 신청 기간·서류·제출 방식은 장학공지의 개별 공지를 함께 확인해 주세요.",
            "이동하시려면 아래 링크를 눌러주세요.",
        ),
    ),
    GeneralGuideTarget(
        label="학사일정",
        url="https://www.hansung.ac.kr/hansung/6096/subview.do",
        aliases=("학사일정", "학사 일정", "일정"),
        guide_terms=("확인", "어디서", "어떻게", "안내", "페이지"),
        notice_exclusion_terms=("공지", "마감", "결과", "발표"),
        answer_lines=(
            "학사일정은 학사일정 페이지에서 확인할 수 있습니다.",
            "학기별 세부 일정은 변경될 수 있으니 공식 페이지 기준으로 확인해 주세요.",
            "이동하시려면 아래 링크를 눌러주세요.",
        ),
    ),
    GeneralGuideTarget(
        label="종합정보시스템",
        url="https://info.hansung.ac.kr/",
        aliases=("종합정보시스템", "종정시", "종합정보"),
        guide_terms=("이용", "방법", "어떻게", "어디서", "안내", "접속"),
        notice_exclusion_terms=("공지", "마감", "결과", "발표"),
        answer_lines=(
            "종합정보시스템은 수강·학적 등 학교 행정 서비스를 확인할 때 사용하는 공식 시스템입니다.",
            "세부 메뉴와 신청 가능 항목은 로그인 후 시스템에서 확인해 주세요.",
            "이동하시려면 아래 링크를 눌러주세요.",
        ),
    ),
    GeneralGuideTarget(
        label="학술정보관",
        url="https://hsel.hansung.ac.kr/",
        aliases=("학술정보관", "학정관", "도서관", "hsel"),
        guide_terms=("이용", "방법", "어떻게", "어디서", "안내", "운영", "시간"),
        notice_exclusion_terms=("공지", "마감", "결과", "발표"),
        answer_lines=(
            "학술정보관 이용 안내는 학술정보관 공식 페이지에서 확인할 수 있습니다.",
            "운영시간, 대출·반납, 시설 이용 정보는 공식 페이지 기준으로 확인해 주세요.",
            "이동하시려면 아래 링크를 눌러주세요.",
        ),
    ),
)


def build_general_guide_response(message: str) -> MainChatResponse | None:
    normalized = _compact(message)
    if not normalized:
        return None
    for target in GENERAL_GUIDE_TARGETS:
        if _matches_general_guide_target(normalized, target):
            return _build_response(message, target)
    return None


def _matches_general_guide_target(normalized_message: str, target: GeneralGuideTarget) -> bool:
    if not any(_compact(alias) in normalized_message for alias in target.aliases):
        return False
    if any(_compact(term) in normalized_message for term in target.notice_exclusion_terms):
        return False
    return any(_compact(term) in normalized_message for term in target.guide_terms)


def _build_response(message: str, target: GeneralGuideTarget) -> MainChatResponse:
    source = MainSource(
        title=target.label,
        url=target.url,
        documentId="official-guide",
        chunkId=_compact(target.label) or "official-guide",
        category="official-guide",
        postedDate=None,
        score=1.0,
    )
    return MainChatResponse(
        intent="MAIN_GENERAL",
        answer="\n".join((*target.answer_lines, target.url)),
        sources=[source],
        confidence=1.0,
        fallbackUsed=False,
        fallbackReason=None,
        searchKeyword=message.strip(),
        resultCount=1,
    )


def _compact(value: str) -> str:
    return "".join((value or "").lower().split())
