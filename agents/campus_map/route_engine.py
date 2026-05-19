from __future__ import annotations

import heapq
from dataclasses import dataclass
from typing import Any

from agents.campus_map.profile import RouteProfile, edge_allowed, edge_cost
from agents.campus_map.repository import CampusMapRepository


@dataclass(frozen=True)
class RoutePathResult:
    path_nodes: list[dict[str, Any]]
    path_edges: list[dict[str, Any]]
    distance_m: float


def shortest_path(
    repo: CampusMapRepository,
    start_id: str,
    end_id: str,
    *,
    profile: RouteProfile,
) -> RoutePathResult | None:
    graph: dict[str, list[tuple[str, float, dict[str, Any]]]] = {}
    for edge in repo.path_edges:
        if not edge_allowed(edge, profile):
            continue
        cost = edge_cost(edge, profile)
        graph.setdefault(edge["from"], []).append((edge["to"], cost, edge))
        graph.setdefault(edge["to"], []).append((edge["from"], cost, edge))

    queue: list[tuple[float, str]] = [(0.0, start_id)]
    distances = {start_id: 0.0}
    previous: dict[str, tuple[str, dict[str, Any]]] = {}
    while queue:
        distance, node_id = heapq.heappop(queue)
        if node_id == end_id:
            break
        if distance > distances.get(node_id, float("inf")):
            continue
        for next_id, edge_distance, edge in graph.get(node_id, []):
            next_distance = distance + edge_distance
            if next_distance < distances.get(next_id, float("inf")):
                distances[next_id] = next_distance
                previous[next_id] = (node_id, edge)
                heapq.heappush(queue, (next_distance, next_id))

    if end_id not in distances:
        return None

    node_ids = [end_id]
    edge_path: list[dict[str, Any]] = []
    while node_ids[-1] != start_id:
        prev_node_id, edge = previous[node_ids[-1]]
        edge_path.append(edge)
        node_ids.append(prev_node_id)
    node_ids.reverse()
    edge_path.reverse()
    nodes = [repo.node(node_id) for node_id in node_ids]
    return RoutePathResult(
        path_nodes=[node for node in nodes if node is not None],
        path_edges=edge_path,
        distance_m=distances[end_id],
    )
