from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Optional


@dataclass
class AgentConfig:
    """
    Agent 默认配置。

    说明：
    - 你没有真实经纬度映射时，前端必须有一个“演示坐标/范围”。
      这里把默认中心从“北京城区占位”改成一个更像农田区域的 demo 坐标（可自行替换）。
    - 一旦你提供 field_meta bbox/geojson，本 Agent 会自动用 bbox 的中心作为镜头中心，不再依赖默认值。
    """

    # Demo center (replace with your own demo farmland location if you like)
    # Example: Shandong farmland area (demo only)
    default_center_lnglat: Tuple[float, float] = (116.35, 37.45)

    # Demo extent (bigger => less "city block", more "field" feeling)
    default_extent_deg: float = 0.02

    # Severity thresholds (percentiles of "low vigor" score)
    mild_pctl: float = 0.70
    moderate_pctl: float = 0.85
    severe_pctl: float = 0.93

    # sampling points
    max_sampling_points: int = 10

    # report
    report_title_cn: str = "作物营养缺乏（Nutrient Deficiency）诊断与处置建议报告"
    report_title_en: str = "Nutrient Deficiency Diagnosis & Action Report"


@dataclass
class LLMConfig:
    mode: str = "off"  # off / qwen2.5 / qwen3-vl / auto
    qwen25_path: Optional[Path] = None
    qwen3vl_path: Optional[Path] = None
    max_new_tokens: int = 900
    temperature: float = 0.35
    top_p: float = 0.9
    repetition_penalty: float = 1.05