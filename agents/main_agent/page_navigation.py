from __future__ import annotations

from dataclasses import dataclass

from agents.main_agent.api.schemas import MainChatResponse, MainSource

_NAVIGATION_TERMS = (
    "이동",
    "열어",
    "열어줘",
    "바로가기",
    "링크",
    "페이지",
    "사이트",
    "접속",
)
_INFORMATION_QUESTION_TERMS = (
    "몇시",
    "몇 시",
    "언제",
    "운영시간",
    "운영 시간",
    "개관",
    "휴관",
    "몇시에",
    "몇 시에",
    "몇시까지",
    "몇 시까지",
    "알려줘",
)
_EXPLICIT_PAGE_TERMS = (
    "페이지",
    "사이트",
    "링크",
    "바로가기",
    "이동",
    "접속",
)


@dataclass(frozen=True)
class PageNavigationTarget:
    label: str
    url: str
    aliases: tuple[str, ...]


PAGE_NAVIGATION_TARGETS: tuple[PageNavigationTarget, ...] = (
    PageNavigationTarget("웹메일", "https://mail.hansung.ac.kr/", ("웹메일", "메일", "한성 웹메일")),
    PageNavigationTarget("전자 조달", "https://www.ebiz4u.co.kr/home.do", ("전자조달", "전자 조달", "인터넷 전자조달", "조달")),
    PageNavigationTarget("학술정보관", "https://hsel.hansung.ac.kr/", ("학술정보관", "학정관", "도서관", "hsel")),
    PageNavigationTarget("대학 일자리 플러스 센터", "https://career.hansung.ac.kr/", ("대학일자리플러스센터", "대학 일자리 플러스 센터", "일자리센터", "일자리 플러스", "커리어")),
    PageNavigationTarget("제증명 발급", "https://career.hansung.ac.kr/", ("제증명", "제증명발급", "증명서", "증명서 발급")),
    PageNavigationTarget("공지사항", "https://www.hansung.ac.kr/hansung/6172/subview.do", ("공지사항", "공지", "한성공지")),
    PageNavigationTarget("eclass", "https://learn.hansung.ac.kr/login.php?errorcode=4", ("eclass", "이클래스", "e-class", "학습관리")),
    PageNavigationTarget("종합정보시스템", "https://info.hansung.ac.kr/", ("종합정보시스템", "종정시", "수강신청", "종합정보", "info")),
    PageNavigationTarget("장학금 안내", "https://hansung.ac.kr/edubank/5762/subview.do", ("장학금", "장학금 안내", "장학")),
    PageNavigationTarget("공간예약", "https://www.hansung.ac.kr/cncschool/4182/subview.do?enc=Zm5jdDF8QEB8JTJGcmVzdmUlMkZjbmNzY2hvb2wlMkY3JTJGYXJ0Y2xSZWdpc3RWaWV3LmRvJTNG", ("공간예약", "공간 예약", "시설예약", "시설 예약")),
    PageNavigationTarget("학사일정", "https://www.hansung.ac.kr/hansung/6096/subview.do", ("학사일정", "학사 일정", "일정")),
    PageNavigationTarget("연구정보", "https://hansung.ac.kr/sites/rnd/index.do", ("연구정보", "연구 정보", "산학연구", "연구")),
)


def resolve_page_navigation_target(message: str) -> PageNavigationTarget | None:
    normalized = _compact(message)
    if not any(_compact(term) in normalized for term in _NAVIGATION_TERMS):
        return None
    if any(_compact(term) in normalized for term in _INFORMATION_QUESTION_TERMS):
        return None
    if not (
        any(_compact(term) in normalized for term in _EXPLICIT_PAGE_TERMS)
        or normalized.endswith("열어")
        or normalized.endswith("열어줘")
    ):
        return None
    for target in PAGE_NAVIGATION_TARGETS:
        if any(_compact(alias) in normalized for alias in target.aliases):
            return target
    return None


def is_page_navigation_request(message: str) -> bool:
    return resolve_page_navigation_target(message) is not None


def build_page_navigation_response(message: str) -> MainChatResponse | None:
    target = resolve_page_navigation_target(message)
    if target is None:
        return None
    source = MainSource(
        title=target.label,
        url=target.url,
        documentId="quick-menu",
        chunkId=_compact(target.label) or "page-navigation",
        category="quick-menu",
        postedDate=None,
        score=1.0,
    )
    return MainChatResponse(
        intent="MAIN_GENERAL",
        answer="\n".join(
            [
                f"{target.label} 페이지로 이동할 수 있는 링크입니다.",
                "이동하시려면 아래 링크를 눌러주세요.",
                target.url,
            ]
        ),
        sources=[source],
        confidence=1.0,
        fallbackUsed=False,
        fallbackReason=None,
        searchKeyword=message.strip(),
        resultCount=1,
    )


def _compact(value: str) -> str:
    return "".join((value or "").lower().split())
