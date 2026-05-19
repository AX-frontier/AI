from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


CampusMapIntent = Literal[
    "PLACE_LOOKUP",
    "ROUTE_GUIDANCE",
    "ENTRANCE_INFO",
    "AMBIGUOUS_PLACE",
    "NO_PLACE_FOUND",
]


class ClientLocation(BaseModel):
    latitude: float | None = None
    longitude: float | None = None
    lat: float | None = None
    lng: float | None = None
    accuracy: float | None = None
    x: float | None = None
    y: float | None = None


class CampusMapRequest(BaseModel):
    queryUid: str
    traceId: str
    conversationUid: str
    message: str = Field(min_length=1)
    clientLocation: ClientLocation | dict[str, Any] | None = None


class MapPoint(BaseModel):
    x: float
    y: float
    lat: float | None = None
    lng: float | None = None


class CampusPlace(BaseModel):
    id: str
    name: str
    aliases: list[str] = Field(default_factory=list)
    category: str
    center: MapPoint
    polygon: list[MapPoint] = Field(default_factory=list)
    representativeFacilities: list[str] = Field(default_factory=list)
    height: float | None = None
    floorCount: int | None = None
    minHeight: float | None = None
    entranceLevel: int | None = None
    geometry: dict[str, Any] = Field(default_factory=dict)
    render3d: dict[str, Any] = Field(default_factory=dict)
    dataQuality: dict[str, Any] = Field(default_factory=dict)


class CampusEntrance(BaseModel):
    id: str
    placeId: str
    name: str
    x: float
    y: float
    accessible: bool
    nodeId: str
    description: str | None = None


class CampusRoute(BaseModel):
    distanceMeters: int
    etaMinutes: int
    accessible: bool
    path: list[MapPoint] = Field(default_factory=list)
    steps: list[str] = Field(default_factory=list)
    originId: str | None = None
    destinationId: str | None = None
    profile: str = "default"
    pathNodeIds: list[str] = Field(default_factory=list)
    pathEdgeIds: list[str] = Field(default_factory=list)
    geometry: dict[str, Any] = Field(default_factory=dict)


class CampusPathNode(BaseModel):
    id: str
    name: str | None = None
    x: float
    y: float
    lat: float | None = None
    lng: float | None = None
    building: str | None = None
    floor: int | None = None
    z: float | None = None
    type: str = "intersection"


class CampusPathEdge(BaseModel):
    id: str | None = None
    from_: str = Field(alias="from")
    to: str
    geometry: list[MapPoint] = Field(default_factory=list)
    distance: float | None = None
    length_m: float | None = None
    type: str = "walkway"
    accessible: bool = True
    indoor: bool = False
    floor_from: int | None = None
    floor_to: int | None = None


class CampusMapCanvas(BaseModel):
    width: int
    height: int


class CampusMapResult(BaseModel):
    campusId: str = "hansung"
    schemaVersion: str = "2.0"
    mode: Literal["place", "route", "entrance", "candidates", "none"]
    viewCapabilities: list[str] = Field(default_factory=lambda: ["svg2d", "deck3d"])
    coordinateSystem: dict[str, Any] = Field(default_factory=dict)
    canvas: CampusMapCanvas
    places: list[CampusPlace] = Field(default_factory=list)
    selectedPlace: CampusPlace | None = None
    entrance: CampusEntrance | None = None
    route: CampusRoute | None = None
    candidates: list[CampusPlace] = Field(default_factory=list)
    pathNodes: list[CampusPathNode] = Field(default_factory=list)
    pathEdges: list[CampusPathEdge] = Field(default_factory=list)


class CampusMapResponse(BaseModel):
    targetAgent: Literal["CAMPUS_MAP"] = "CAMPUS_MAP"
    intent: CampusMapIntent
    answer: str
    sources: list[dict[str, Any]] = Field(default_factory=list)
    confidence: float
    fallbackUsed: bool
    fallbackReason: str | None = None
    searchKeyword: str | None = None
    resultCount: int = 0
    mapResult: CampusMapResult
