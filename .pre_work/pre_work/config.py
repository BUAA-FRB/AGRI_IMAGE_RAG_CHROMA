from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

@dataclass
class SiteConfig:
    name: str
    lat: float
    lon: float
    crop: str = "unknown"
    stage: str = "unknown"
    timezone: str = "auto"

@dataclass
class RunConfig:
    days_hist: int = 14
    days_fore: int = 16
    extend_to_30: bool = False

    open_meteo_model: str = "best_match"
    use_soilgrids: bool = True
    cache_dir: str = ".pre_work/.cache"
    cache_ttl_hours: int = 12

    out_root: str = ".pre_work_out"

    use_llm: bool = True
    model_path: str = "./models/Qwen/Qwen2.5-3B-Instruct"
    device: str = "auto"
    dtype: str = "auto"
    max_new_tokens: int = 1800
    temperature: float = 0.35
    top_p: float = 0.85
    repetition_penalty: float = 1.05
    seed: Optional[int] = 42

    frost_c: float = 0.0
    heat_c: float = 35.0
    heavy_rain_mm_day: float = 50.0
    dry_spell_days: int = 7
