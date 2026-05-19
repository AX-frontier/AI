from __future__ import annotations

from typing import Any


def coordinate_system(data: dict[str, Any]) -> dict[str, Any]:
    canvas = data.get("canvas") or {"width": 840, "height": 1040}
    return data.get("coordinateSystem") or {
        "type": "LOCAL_METER_WITH_CANVAS_AND_WGS84",
        "origin": {"lat": 37.58255, "lng": 127.01015, "x": 0, "y": 0, "z": 0},
        "axes": {"x": "east_m", "y": "north_m", "z": "up_m"},
        "canvas": canvas,
        "canvasToLocal": {
            "scaleX_mPerPx": 0.5,
            "scaleY_mPerPx": 0.5,
            "originCanvasX": canvas["width"] / 2,
            "originCanvasY": canvas["height"] / 2,
            "flipY": True,
        },
    }


def canvas_to_local(point: dict[str, Any], coordinate_system_data: dict[str, Any]) -> list[float]:
    transform = coordinate_system_data["canvasToLocal"]
    origin_x = float(transform["originCanvasX"])
    origin_y = float(transform["originCanvasY"])
    scale_x = float(transform["scaleX_mPerPx"])
    scale_y = float(transform["scaleY_mPerPx"])
    flip_y = bool(transform.get("flipY", True))
    x = (float(point["x"]) - origin_x) * scale_x
    y_direction = origin_y - float(point["y"]) if flip_y else float(point["y"]) - origin_y
    return [round(x, 3), round(y_direction * scale_y, 3)]


def local3d_from_canvas(point: dict[str, Any], coordinate_system_data: dict[str, Any], *, z: float = 0.2) -> list[float]:
    x, y = canvas_to_local(point, coordinate_system_data)
    return [x, y, z]


def build_route_geometry(
    path_nodes: list[dict[str, Any]],
    coordinate_system_data: dict[str, Any],
) -> dict[str, list[list[float]]]:
    canvas_points = [[float(node["x"]), float(node["y"])] for node in path_nodes]
    local2d = [canvas_to_local({"x": node["x"], "y": node["y"]}, coordinate_system_data) for node in path_nodes]
    local3d = [local3d_from_canvas({"x": node["x"], "y": node["y"]}, coordinate_system_data) for node in path_nodes]
    geo2d = [
        [float(node["lng"]), float(node["lat"])]
        for node in path_nodes
        if node.get("lat") is not None and node.get("lng") is not None
    ]
    geo3d = [[lng, lat, 0.2] for lng, lat in geo2d]
    return {
        "canvas": canvas_points,
        "local2d": local2d,
        "local3d": local3d,
        "geo2d": geo2d,
        "geo3d": geo3d,
    }


def enrich_place_for_v2(place: dict[str, Any], coordinate_system_data: dict[str, Any]) -> dict[str, Any]:
    result = dict(place)
    center = place.get("center") or {"x": 0, "y": 0}
    polygon = place.get("polygon") or []
    height = place.get("height_m") or place.get("height") or 18
    floor_count = place.get("floor_count") or place.get("floorCount") or 5
    min_height = place.get("min_height_m") or place.get("minHeight") or 0
    entrance_level = place.get("entrance_level") or place.get("entranceLevel") or 1
    result["geometry"] = {
        "centerCanvas": center,
        "polygonCanvas": [[point["x"], point["y"]] for point in polygon],
        "centerLocal": {
            "x": canvas_to_local(center, coordinate_system_data)[0],
            "y": canvas_to_local(center, coordinate_system_data)[1],
            "z": 0,
        },
        "footprintLocal": [canvas_to_local(point, coordinate_system_data) for point in polygon],
        "centerGeo": {"lat": center.get("lat"), "lng": center.get("lng")},
        "footprintGeo": None,
    }
    result["render3d"] = {
        "renderType": "extrusion",
        "height_m": height,
        "base_z_m": 0,
        "min_height_m": min_height,
        "floor_count": floor_count,
        "floor_height_m": round(float(height) / max(1, int(floor_count)), 2),
        "entrance_level": entrance_level,
        "roof_type": "flat",
        "model_url": None,
    }
    result["dataQuality"] = place.get("dataQuality") or {
        "source": "manual_seed",
        "accuracy": "mock",
        "verified": False,
    }
    return result
