from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any

@dataclass
class SoilProfile:
    clay: Optional[float] = None
    sand: Optional[float] = None
    silt: Optional[float] = None
    oc: Optional[float] = None
    bdod: Optional[float] = None
    source: str = "soilgrids"

@dataclass
class WeatherFrame:
    daily: Dict[str, List[Any]] = field(default_factory=dict)
    hourly: Dict[str, List[Any]] = field(default_factory=dict)
    meta: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Indicators:
    daily: Dict[str, List[float]] = field(default_factory=dict)
    meta: Dict[str, Any] = field(default_factory=dict)

@dataclass
class RiskOutput:
    daily_risk: Dict[str, List[float]] = field(default_factory=dict)
    daily_level: Dict[str, List[str]] = field(default_factory=dict)
    triggers: Dict[str, List[str]] = field(default_factory=dict)
    summary: Dict[str, Any] = field(default_factory=dict)
