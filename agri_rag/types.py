
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any

BBox = Tuple[int, int, int, int]  # (x1, y1, x2, y2)

@dataclass
class TileRecord:
    tile_id: str
    split: Optional[str]
    rgb_path: str
    nir_path: Optional[str]
    field_mask_path: Optional[str]
    labels_present: List[str]
    label_areas: Dict[str, int]
    extra: Dict[str, Any]

@dataclass
class RetrievalHit:
    id: str
    distance: float
    metadata: Dict[str, Any]
    document: Optional[str] = None
