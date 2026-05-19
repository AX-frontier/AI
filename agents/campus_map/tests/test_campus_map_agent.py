from __future__ import annotations

from agents.campus_map.agent import run_campus_map_agent
from agents.campus_map.api.schemas import CampusMapRequest
from agents.campus_map.repository import get_campus_map_repository


def make_request(message: str, client_location: dict | None = None) -> CampusMapRequest:
    return CampusMapRequest(
        queryUid="query-1",
        traceId="trace-1",
        conversationUid="conversation-1",
        message=message,
        clientLocation=client_location,
    )


def test_typo_search_resolves_student_union() -> None:
    response = run_campus_map_agent(make_request("학생회간 위치 알려줘"))

    assert response.intent == "PLACE_LOOKUP"
    assert response.mapResult.selectedPlace is not None
    assert response.mapResult.selectedPlace.name == "학생회관"


def test_library_location_resolves_to_campus_library() -> None:
    response = run_campus_map_agent(make_request("도서관 어디야"))

    assert response.intent == "PLACE_LOOKUP"
    assert response.mapResult.selectedPlace is not None
    assert response.mapResult.selectedPlace.name == "학술정보관"


def test_engineering_hall_returns_ambiguous_candidates() -> None:
    response = run_campus_map_agent(make_request("공학관 어디야"))

    assert response.intent == "AMBIGUOUS_PLACE"
    assert {place.name for place in response.mapResult.candidates} >= {"공학관A동", "공학관B동"}


def test_route_from_library_to_student_union() -> None:
    response = run_campus_map_agent(make_request("학술정보관에서 학생회관 가는 길"))

    assert response.intent == "ROUTE_GUIDANCE"
    assert response.mapResult.route is not None
    assert response.mapResult.route.distanceMeters > 0
    assert response.mapResult.route.path


def test_accessible_route_avoids_inaccessible_edges() -> None:
    response = run_campus_map_agent(make_request("정문에서 제1공학관까지 휠체어로 가는 길"))

    assert response.intent == "ROUTE_GUIDANCE"
    assert response.mapResult.route is not None
    assert response.mapResult.route.accessible is True


def test_client_location_is_accepted_for_current_location_route() -> None:
    repo = get_campus_map_repository()
    location = {"latitude": 37.58225, "longitude": 127.0112, "accuracy": 20}

    response = run_campus_map_agent(
        make_request("내 위치에서 상상관까지 어떻게 가?", location),
        repository=repo,
    )

    assert response.intent == "ROUTE_GUIDANCE"
    assert response.mapResult.route is not None
    assert response.mapResult.route.etaMinutes >= 1
