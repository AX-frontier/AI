from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from agents.campus_map.geometry_adapter import build_route_geometry, enrich_place_for_v2
from agents.campus_map.profile import RouteProfile, infer_route_profile
from agents.campus_map.route_engine import shortest_path
from agents.campus_map.api.schemas import (
    CampusEntrance,
    CampusMapRequest,
    CampusMapResponse,
    CampusMapResult,
    CampusPathEdge,
    CampusPathNode,
    CampusPlace,
    CampusRoute,
    MapPoint,
)
from agents.campus_map.repository import CampusMapRepository, SearchMatch, get_campus_map_repository


_ROUTE_TERMS = ("가는 길", "가는길", "길찾기", "어떻게 가", "까지", "에서", "내 위치", "현재 위치")
_ENTRANCE_TERMS = ("출입구", "정문", "후문", "입구")


@dataclass(frozen=True)
class RouteRequestParts:
    origin_keyword: str | None
    destination_keyword: str
    use_client_location: bool


def run_campus_map_agent(
    request: CampusMapRequest,
    *,
    repository: CampusMapRepository | None = None,
) -> CampusMapResponse:
    repo = repository or get_campus_map_repository()
    message = request.message.strip()
    profile = infer_route_profile(message)
    accessible_only = profile in ("accessible", "avoid_stairs", "elevator_required")
    route_requested = _looks_route_query(message)
    entrance_requested = any(term in message for term in _ENTRANCE_TERMS)

    parts = _parse_route_parts(message)
    destination_matches = repo.search_places(parts.destination_keyword, limit=5)
    if not destination_matches:
        return _no_place_response(repo, parts.destination_keyword)
    if _is_ambiguous(destination_matches, parts.destination_keyword):
        return _ambiguous_response(repo, parts.destination_keyword, destination_matches)

    destination = destination_matches[0].place
    entrances = repo.entrances_for_place(destination["id"], accessible_only=accessible_only)
    entrance = entrances[0] if entrances else None

    if route_requested:
        route = _build_route(
            repo,
            request,
            parts,
            destination,
            entrance,
            profile=profile,
        )
        return _route_response(repo, destination, entrance, route, parts, accessible_only)
    if entrance_requested:
        return _entrance_response(repo, destination, entrance)
    return _place_response(repo, destination, entrance)


def _looks_route_query(message: str) -> bool:
    return any(term in message for term in _ROUTE_TERMS)


def _parse_route_parts(message: str) -> RouteRequestParts:
    text = message.strip()
    use_client_location = "내 위치" in text or "현재 위치" in text
    origin_keyword: str | None = None
    destination_keyword = text

    if "에서" in text:
        before, after = text.split("에서", 1)
        origin_keyword = before.replace("내 위치", "").replace("현재 위치", "").strip() or None
        destination_keyword = after
    for token in ("까지", "가는 길", "가는길", "길찾기", "어떻게 가", "가려면", "알려줘", "위치", "어디야", "어디 있어", "휠체어로", "휠체어", "계단 없이", "?"):
        destination_keyword = destination_keyword.replace(token, " ")
    destination_keyword = " ".join(destination_keyword.split()) or text
    return RouteRequestParts(origin_keyword=origin_keyword, destination_keyword=destination_keyword, use_client_location=use_client_location)


def _is_ambiguous(matches: list[SearchMatch], keyword: str) -> bool:
    if len(matches) < 2:
        return False
    top = matches[0]
    second = matches[1]
    if top.score >= 1.0 and second.score >= 1.0:
        return True
    return top.score - second.score < 0.04 and second.score >= 0.75


def _build_route(
    repo: CampusMapRepository,
    request: CampusMapRequest,
    parts: RouteRequestParts,
    destination: dict[str, Any],
    entrance: dict[str, Any] | None,
    *,
    profile: RouteProfile,
) -> CampusRoute | None:
    destination_node = repo.node(entrance["nodeId"]) if entrance else None
    if destination_node is None:
        return None

    origin_node = None
    if parts.use_client_location:
        point = repo.point_from_client_location(request.clientLocation)
        if point is not None:
            origin_node = repo.nearest_node_for_point(point, accessible_only=profile == "accessible")
    if origin_node is None and parts.origin_keyword:
        origin_matches = repo.search_places(parts.origin_keyword, limit=1)
        if origin_matches:
            origin_entrances = repo.entrances_for_place(origin_matches[0].place["id"], accessible_only=profile == "accessible")
            if origin_entrances:
                origin_node = repo.node(origin_entrances[0]["nodeId"])
    if origin_node is None:
        origin_node = repo.node("n-main-gate")
    if origin_node is None:
        return None

    route_result = shortest_path(repo, origin_node["id"], destination_node["id"], profile=profile)
    if route_result is None or not route_result.path_nodes:
        return None
    path_nodes = route_result.path_nodes
    path_edges = route_result.path_edges
    distance = route_result.distance_m
    path = [MapPoint(x=float(node["x"]), y=float(node["y"]), lat=node.get("lat"), lng=node.get("lng")) for node in path_nodes]
    eta = max(1, round(distance / 70))
    steps = _route_steps(origin_node, destination, entrance, distance, profile == "accessible")
    return CampusRoute(
        distanceMeters=int(round(distance)),
        etaMinutes=int(eta),
        accessible=all(_edge_accessible(repo, path_nodes[i]["id"], path_nodes[i + 1]["id"]) for i in range(len(path_nodes) - 1)),
        path=path,
        steps=steps,
        originId=origin_node["id"],
        destinationId=destination_node["id"],
        profile=profile,
        pathNodeIds=[node["id"] for node in path_nodes],
        pathEdgeIds=[str(edge.get("id") or f"{edge['from']}->{edge['to']}") for edge in path_edges],
        geometry=build_route_geometry(path_nodes, repo.coordinate_system),
    )


def _edge_accessible(repo: CampusMapRepository, left: str, right: str) -> bool:
    for edge in repo.path_edges:
        if {edge["from"], edge["to"]} == {left, right}:
            return bool(edge.get("accessible", True))
    return True


def _route_steps(
    origin_node: dict[str, Any],
    destination: dict[str, Any],
    entrance: dict[str, Any] | None,
    distance: float,
    accessible_only: bool,
) -> list[str]:
    start = "현재 위치와 가까운 보행로" if origin_node["id"] != "n-main-gate" else "정문"
    entrance_name = entrance["name"] if entrance else "대표 입구"
    steps = [
        f"{start}에서 캠퍼스 보행로를 따라 이동합니다.",
        f"{destination['name']} {entrance_name} 방향으로 진입합니다.",
        f"예상 거리는 약 {int(round(distance))}m입니다.",
    ]
    if accessible_only:
        steps.insert(1, "계단 구간을 피하는 접근성 우선 경로를 사용합니다.")
    return steps


def _place_response(repo: CampusMapRepository, place: dict[str, Any], entrance: dict[str, Any] | None) -> CampusMapResponse:
    facilities = _facility_text(place)
    entrance_text = f" 대표 출입구는 {entrance['name']}입니다." if entrance else ""
    return CampusMapResponse(
        intent="PLACE_LOOKUP",
        answer=f"{place['name']}은(는) 캠퍼스 2D 맵에 표시된 위치에 있습니다.{entrance_text}{facilities}",
        confidence=0.9,
        fallbackUsed=False,
        searchKeyword=place["name"],
        resultCount=1,
        mapResult=_map_result(repo, "place", selected_place=place, entrance=entrance),
    )


def _entrance_response(repo: CampusMapRepository, place: dict[str, Any], entrance: dict[str, Any] | None) -> CampusMapResponse:
    if entrance:
        answer = f"{place['name']}의 대표 출입구는 {entrance['name']}입니다. {entrance.get('description') or ''}".strip()
    else:
        answer = f"{place['name']}의 출입구 데이터가 아직 준비되지 않았습니다."
    return CampusMapResponse(
        intent="ENTRANCE_INFO",
        answer=answer,
        confidence=0.88 if entrance else 0.55,
        fallbackUsed=entrance is None,
        fallbackReason=None if entrance else "ENTRANCE_NOT_FOUND",
        searchKeyword=place["name"],
        resultCount=1,
        mapResult=_map_result(repo, "entrance", selected_place=place, entrance=entrance),
    )


def _route_response(
    repo: CampusMapRepository,
    place: dict[str, Any],
    entrance: dict[str, Any] | None,
    route: CampusRoute | None,
    parts: RouteRequestParts,
    accessible_only: bool,
) -> CampusMapResponse:
    if route is None:
        return CampusMapResponse(
            intent="ROUTE_GUIDANCE",
            answer=f"{place['name']} 위치는 찾았지만 연결된 보행 경로 데이터가 아직 부족합니다.",
            confidence=0.58,
            fallbackUsed=True,
            fallbackReason="ROUTE_NOT_FOUND",
            searchKeyword=parts.destination_keyword,
            resultCount=1,
            mapResult=_map_result(repo, "place", selected_place=place, entrance=entrance),
        )
    origin_label = "현재 위치" if parts.use_client_location else (parts.origin_keyword or "정문")
    accessible_label = " 접근성 우선 경로입니다." if accessible_only and route.accessible else ""
    answer = (
        f"{origin_label}에서 {place['name']} {entrance['name'] if entrance else '대표 입구'}까지 "
        f"약 {route.distanceMeters}m, 도보 {route.etaMinutes}분 거리입니다.{accessible_label}"
    )
    return CampusMapResponse(
        intent="ROUTE_GUIDANCE",
        answer=answer,
        confidence=0.91,
        fallbackUsed=False,
        searchKeyword=parts.destination_keyword,
        resultCount=1,
        mapResult=_map_result(repo, "route", selected_place=place, entrance=entrance, route=route),
    )


def _ambiguous_response(repo: CampusMapRepository, keyword: str, matches: list[SearchMatch]) -> CampusMapResponse:
    candidates = [match.place for match in matches[:4]]
    names = ", ".join(place["name"] for place in candidates)
    return CampusMapResponse(
        intent="AMBIGUOUS_PLACE",
        answer=f'"{keyword}"에 해당하는 후보가 여러 곳입니다. {names} 중 어느 곳인지 다시 입력해 주세요.',
        confidence=0.64,
        fallbackUsed=False,
        searchKeyword=keyword,
        resultCount=len(candidates),
        mapResult=_map_result(repo, "candidates", candidates=candidates),
    )


def _no_place_response(repo: CampusMapRepository, keyword: str) -> CampusMapResponse:
    return CampusMapResponse(
        intent="NO_PLACE_FOUND",
        answer=f'"{keyword}"에 해당하는 캠퍼스 장소를 아직 찾지 못했습니다. 건물명이나 별칭을 조금 더 구체적으로 입력해 주세요.',
        confidence=0.35,
        fallbackUsed=True,
        fallbackReason="NO_PLACE_FOUND",
        searchKeyword=keyword,
        resultCount=0,
        mapResult=_map_result(repo, "none"),
    )


def _map_result(
    repo: CampusMapRepository,
    mode: str,
    *,
    selected_place: dict[str, Any] | None = None,
    entrance: dict[str, Any] | None = None,
    route: CampusRoute | None = None,
    candidates: list[dict[str, Any]] | None = None,
) -> CampusMapResult:
    return CampusMapResult(
        campusId=repo.campus_id,
        schemaVersion=repo.schema_version,
        mode=mode,
        viewCapabilities=["svg2d", "deck3d"],
        coordinateSystem=repo.coordinate_system,
        canvas=repo.canvas,
        places=[CampusPlace.model_validate(enrich_place_for_v2(place, repo.coordinate_system)) for place in repo.places],
        selectedPlace=CampusPlace.model_validate(enrich_place_for_v2(selected_place, repo.coordinate_system)) if selected_place else None,
        entrance=CampusEntrance.model_validate(entrance) if entrance else None,
        route=route,
        candidates=[CampusPlace.model_validate(place) for place in (candidates or [])],
        pathNodes=[CampusPathNode.model_validate(node) for node in repo.path_nodes],
        pathEdges=[CampusPathEdge.model_validate(edge) for edge in repo.path_edges],
    )


def _facility_text(place: dict[str, Any]) -> str:
    facilities = place.get("representativeFacilities") or []
    if not facilities:
        return ""
    return " 주요 시설은 " + ", ".join(facilities[:3]) + "입니다."
