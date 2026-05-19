from __future__ import annotations

from typing import Any, Literal


RouteProfile = Literal[
    "default",
    "accessible",
    "indoor_preferred",
    "avoid_slope",
    "avoid_stairs",
    "elevator_required",
    "safe_night",
]


_ACCESSIBLE_TERMS = ("휠체어", "엘리베이터", "계단 없이", "장애", "무장애")
_INDOOR_TERMS = ("실내", "건물 안", "비 안 맞", "비맞")
_SLOPE_TERMS = ("경사", "언덕", "오르막")
_NIGHT_TERMS = ("밤", "야간", "어두")


def infer_route_profile(message: str) -> RouteProfile:
    if any(term in message for term in _ACCESSIBLE_TERMS):
        return "accessible"
    if any(term in message for term in _SLOPE_TERMS):
        return "avoid_slope"
    if any(term in message for term in _INDOOR_TERMS):
        return "indoor_preferred"
    if any(term in message for term in _NIGHT_TERMS):
        return "safe_night"
    return "default"


def edge_allowed(edge: dict[str, Any], profile: RouteProfile) -> bool:
    connector_type = str(edge.get("connectorType") or edge.get("type") or "")
    cost_profiles = edge.get("costProfiles") or {}
    profile_cost = cost_profiles.get(profile)
    if profile_cost is None and profile in cost_profiles:
        return False
    if profile in ("accessible", "avoid_stairs", "elevator_required"):
        if not edge.get("accessible", True):
            return False
        if connector_type == "stairs":
            return False
    if profile == "elevator_required" and connector_type in ("stairs", "ramp"):
        return False
    return True


def edge_cost(edge: dict[str, Any], profile: RouteProfile) -> float:
    cost_profiles = edge.get("costProfiles") or {}
    profile_cost = cost_profiles.get(profile)
    if isinstance(profile_cost, int | float):
        return float(profile_cost)
    if profile == "indoor_preferred" and edge.get("indoor"):
        return float(edge.get("length_m") or edge.get("distance") or 1) * 0.85
    return float(edge.get("length_m") or edge.get("distance") or 1)
