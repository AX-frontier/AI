from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from agents.campus_map.geometry_adapter import coordinate_system


_DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "samples" / "campus_map" / "hansung_campus.json"


@dataclass(frozen=True)
class SearchMatch:
    place: dict[str, Any]
    score: float
    matched_text: str


class CampusMapRepository:
    def __init__(self, data: dict[str, Any]):
        self.data = data
        self.places = list(data.get("places", []))
        self.entrances = list(data.get("entrances", []))
        self.path_nodes = list(data.get("pathNodes", []))
        self.path_edges = list(data.get("pathEdges", []))
        self._place_by_id = {place["id"]: place for place in self.places}
        self._entrances_by_place: dict[str, list[dict[str, Any]]] = {}
        for entrance in self.entrances:
            self._entrances_by_place.setdefault(entrance["placeId"], []).append(entrance)
        self._node_by_id = {node["id"]: node for node in self.path_nodes}

    @property
    def campus_id(self) -> str:
        return str(self.data.get("campusId", "hansung"))

    @property
    def canvas(self) -> dict[str, int]:
        canvas = self.data.get("canvas") or {}
        return {"width": int(canvas.get("width", 900)), "height": int(canvas.get("height", 680))}

    @property
    def schema_version(self) -> str:
        return str(self.data.get("schemaVersion", "2.0"))

    @property
    def coordinate_system(self) -> dict[str, Any]:
        return coordinate_system(self.data)

    @property
    def geo_bounds(self) -> dict[str, float]:
        return self.data.get("geoBounds") or {}

    def get_place(self, place_id: str) -> dict[str, Any] | None:
        return self._place_by_id.get(place_id)

    def entrances_for_place(self, place_id: str, *, accessible_only: bool = False) -> list[dict[str, Any]]:
        entrances = list(self._entrances_by_place.get(place_id, []))
        if accessible_only:
            accessible = [entrance for entrance in entrances if entrance.get("accessible")]
            return accessible or entrances
        return entrances

    def node(self, node_id: str) -> dict[str, Any] | None:
        return self._node_by_id.get(node_id)

    def search_places(self, keyword: str, *, limit: int = 5) -> list[SearchMatch]:
        normalized_keyword = normalize_text(keyword)
        if not normalized_keyword:
            return []
        keyword_initials = hangul_initials(normalized_keyword)
        matches: list[SearchMatch] = []
        for place in self.places:
            names = [place.get("name", ""), *place.get("aliases", [])]
            best_score = 0.0
            best_text = ""
            for name in names:
                normalized_name = normalize_text(name)
                if not normalized_name:
                    continue
                score = _name_score(normalized_keyword, normalized_name, keyword_initials)
                if score > best_score:
                    best_score = score
                    best_text = name
            if best_score >= 0.55:
                matches.append(SearchMatch(place=place, score=round(best_score, 3), matched_text=best_text))
        matches.sort(key=lambda match: (-match.score, match.place.get("name", "")))
        return matches[:limit]

    def nearest_node_for_point(self, point: dict[str, float], *, accessible_only: bool = False) -> dict[str, Any] | None:
        candidates = self.path_nodes
        if accessible_only:
            blocked = _blocked_stair_nodes(self.path_edges)
            candidates = [node for node in candidates if node.get("id") not in blocked] or candidates
        if not candidates:
            return None
        return min(candidates, key=lambda node: _distance(point["x"], point["y"], node["x"], node["y"]))

    def point_from_client_location(self, location: Any) -> dict[str, float] | None:
        if not location:
            return None
        record = location if isinstance(location, dict) else location.model_dump()
        x = _float_or_none(record.get("x"))
        y = _float_or_none(record.get("y"))
        if x is not None and y is not None:
            return {"x": x, "y": y}
        lat = _float_or_none(record.get("latitude") if record.get("latitude") is not None else record.get("lat"))
        lng = _float_or_none(record.get("longitude") if record.get("longitude") is not None else record.get("lng"))
        bounds = self.geo_bounds
        if lat is None or lng is None or not bounds:
            return None
        min_lat = float(bounds["minLat"])
        max_lat = float(bounds["maxLat"])
        min_lng = float(bounds["minLng"])
        max_lng = float(bounds["maxLng"])
        canvas = self.canvas
        x = (lng - min_lng) / (max_lng - min_lng) * canvas["width"]
        y = (max_lat - lat) / (max_lat - min_lat) * canvas["height"]
        if x < -100 or y < -100 or x > canvas["width"] + 100 or y > canvas["height"] + 100:
            return None
        return {"x": max(0.0, min(float(canvas["width"]), x)), "y": max(0.0, min(float(canvas["height"]), y))}


@lru_cache(maxsize=1)
def get_campus_map_repository() -> CampusMapRepository:
    with _DATA_PATH.open(encoding="utf-8") as file:
        return CampusMapRepository(json.load(file))


def normalize_text(value: str) -> str:
    return "".join(ch for ch in value.lower().strip() if ch.isalnum() or "가" <= ch <= "힣")


def hangul_initials(value: str) -> str:
    initials = "ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ"
    result: list[str] = []
    for ch in value:
        code = ord(ch)
        if 0xAC00 <= code <= 0xD7A3:
            result.append(initials[(code - 0xAC00) // 588])
        elif ch.isalnum():
            result.append(ch)
    return "".join(result)


def _name_score(keyword: str, name: str, keyword_initials: str) -> float:
    if keyword == name:
        return 1.0
    if name in keyword:
        return 0.96 if len(name) >= 3 else 0.84
    if keyword in name:
        return 0.9
    name_initials = hangul_initials(name)
    if keyword_initials and keyword_initials in name_initials:
        return 0.78
    return _similarity(keyword, name) * 0.92


def _similarity(left: str, right: str) -> float:
    if not left or not right:
        return 0.0
    previous = list(range(len(right) + 1))
    for i, left_ch in enumerate(left, start=1):
        current = [i]
        for j, right_ch in enumerate(right, start=1):
            current.append(min(
                previous[j] + 1,
                current[j - 1] + 1,
                previous[j - 1] + (left_ch != right_ch),
            ))
        previous = current
    distance = previous[-1]
    return 1.0 - distance / max(len(left), len(right))


def _distance(x1: float, y1: float, x2: float, y2: float) -> float:
    return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5


def _blocked_stair_nodes(edges: list[dict[str, Any]]) -> set[str]:
    blocked: set[str] = set()
    for edge in edges:
        if not edge.get("accessible", True):
            blocked.add(str(edge.get("from")))
            blocked.add(str(edge.get("to")))
    return blocked


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
