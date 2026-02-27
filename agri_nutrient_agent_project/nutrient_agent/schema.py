from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
from pydantic import BaseModel, Field


class SuspectedNutrient(BaseModel):
    nutrient: str
    prob: float
    evidence: List[str] = Field(default_factory=list)


class AreaStats(BaseModel):
    affected_area_px: int
    affected_ratio: float
    patch_count: int
    largest_patch_px: int


class LayersOut(BaseModel):
    severity_geojson: str
    sampling_points_geojson: str
    severity_heatmap_png: str
    index_preview_png: str


class DecisionOut(BaseModel):
    is_applicable: bool
    upstream_label_conf: float
    agent_validation_score: float
    uncertainty_notes: List[str] = Field(default_factory=list)


class DiagnosisOut(BaseModel):
    summary_cn: str
    summary_en: str
    suspected_nutrients: List[SuspectedNutrient] = Field(default_factory=list)


class FrontendTerrain(BaseModel):
    enabled: bool = True
    exaggeration: float = 1.2
    dem_tiles_json: str = "https://demotiles.maplibre.org/terrain-tiles/tiles.json"


class FrontendScene(BaseModel):
    center: Tuple[float, float]
    default_zoom: float = 15
    terrain: FrontendTerrain = Field(default_factory=FrontendTerrain)
    bbox: Tuple[float, float, float, float]  # [minLng, minLat, maxLng, maxLat]
    layers: Dict[str, Any] = Field(default_factory=dict)


class NutrientDeficiencyOutput(BaseModel):
    schema: str = "agri_agent.nutrient_deficiency.v1"
    time_utc: str
    run_id: str
    field_id: str
    upstream_path: str
    query_image_path: str
    decision: DecisionOut
    diagnosis: DiagnosisOut
    area_stats: AreaStats
    layers: LayersOut
    report_md: str
    assets: Dict[str, Any] = Field(default_factory=dict)
    frontend_scene: FrontendScene
