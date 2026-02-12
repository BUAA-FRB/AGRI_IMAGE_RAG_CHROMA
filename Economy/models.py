from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional


@dataclass
class EconomyItem:
    label: str
    confidence: float

    affected_area_px: float
    field_area_px: float
    affected_ratio: float

    affected_area_m2: Optional[float]
    field_area_m2: Optional[float]
    affected_area_ha: Optional[float]
    field_area_ha: Optional[float]
    affected_area_mu: Optional[float]
    field_area_mu: Optional[float]

    # expected whole-field yield-loss fraction (0~1) attributable to this hazard (heuristic)
    yield_loss_frac_p50: float
    yield_loss_frac_p10: float
    yield_loss_frac_p90: float

    # money loss attributable (heuristic, may double-count if hazards overlap)
    loss_value_p50: Optional[float]
    loss_value_p10: Optional[float]
    loss_value_p90: Optional[float]

    currency: str
    notes: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class EconomyResult:
    schema: str
    time_utc: str

    latest_json_path: str
    field_stats_path: str

    run_id: str

    tile_id: str
    tile_split: Optional[str]
    tile_year: Optional[str]
    tile_source_path: Optional[str]

    crop: str
    stage: str
    currency: str

    gsd_m_per_px: Optional[float]
    area_mode: str            # "gsd" | "ratio_only"
    normalize_to: str         # "mu" | "ha" (used when ratio_only)

    items: List[EconomyItem]

    # base value of the whole field (money) under current area_mode
    base_field_value_p50: Optional[float]  # e.g., field_ha*value_per_ha (gsd) or per-1mu/per-1ha (ratio_only)

    # total combined loss (NOT sum of items; uses combined yield-loss)
    total_loss_value_p50: Optional[float]
    total_loss_value_p10: Optional[float]
    total_loss_value_p90: Optional[float]

    total_yield_loss_frac_p50: float
    total_yield_loss_frac_p10: float
    total_yield_loss_frac_p90: float

    assumptions: Dict[str, Any]
    warnings: List[str]

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["items"] = [it.to_dict() for it in self.items]
        return d
