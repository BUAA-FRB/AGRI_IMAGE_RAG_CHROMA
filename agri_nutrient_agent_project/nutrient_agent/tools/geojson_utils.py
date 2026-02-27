from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np


@dataclass
class BBoxLngLat:
    min_lng: float
    min_lat: float
    max_lng: float
    max_lat: float


def pixel_to_lnglat(x: float, y: float, W: int, H: int, bbox: BBoxLngLat) -> Tuple[float, float]:
    # x in [0,W), y in [0,H)
    lng = bbox.min_lng + (x / max(1, W)) * (bbox.max_lng - bbox.min_lng)
    lat = bbox.max_lat - (y / max(1, H)) * (bbox.max_lat - bbox.min_lat)
    return float(lng), float(lat)


def bbox_polygon_geojson(bbox: BBoxLngLat) -> Dict[str, Any]:
    coords = [
        [bbox.min_lng, bbox.min_lat],
        [bbox.max_lng, bbox.min_lat],
        [bbox.max_lng, bbox.max_lat],
        [bbox.min_lng, bbox.max_lat],
        [bbox.min_lng, bbox.min_lat],
    ]
    return {
        "type": "FeatureCollection",
        "features": [
            {"type": "Feature", "properties": {"id": "field_bbox"}, "geometry": {"type": "Polygon", "coordinates": [coords]}}
        ],
    }


def rectangles_to_geojson(
    rects_xyxy_area: List[Tuple[int,int,int,int,int]],
    W: int,
    H: int,
    bbox: BBoxLngLat,
    props_extra: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    feats = []
    for i, (x0, y0, x1, y1, area) in enumerate(rects_xyxy_area):
        lng0, lat0 = pixel_to_lnglat(x0, y1, W, H, bbox)
        lng1, lat1 = pixel_to_lnglat(x1, y0, W, H, bbox)
        coords = [
            [lng0, lat0],
            [lng1, lat0],
            [lng1, lat1],
            [lng0, lat1],
            [lng0, lat0],
        ]
        props = {"id": f"patch_{i}", "area_px": int(area)}
        if props_extra:
            props.update(props_extra)
        feats.append({"type": "Feature", "properties": props, "geometry": {"type": "Polygon", "coordinates": [coords]}})
    return {"type": "FeatureCollection", "features": feats}


def sampling_points_geojson(points_px: List[Tuple[int,int,str]], W: int, H: int, bbox: BBoxLngLat) -> Dict[str, Any]:
    feats = []
    for i, (x, y, note) in enumerate(points_px):
        lng, lat = pixel_to_lnglat(x, y, W, H, bbox)
        feats.append({
            "type": "Feature",
            "properties": {"id": f"s{i}", "note": note},
            "geometry": {"type": "Point", "coordinates": [lng, lat]},
        })
    return {"type": "FeatureCollection", "features": feats}
