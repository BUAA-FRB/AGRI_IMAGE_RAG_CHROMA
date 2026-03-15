"""
数据结构定义, 包含：
Agent 分析报告类型 (AgentReport, AnomalyDetail, SpatialAnalysis, etc.)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


# ============================================================
# 基础类型（与原有代码兼容）
# ============================================================

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


# ============================================================
# Agent 分析报告类型
# ============================================================

class SeverityLevel(Enum):
    """异常严重程度等级"""
    NONE = "none"            # 未检测到异常
    MINOR = "minor"          # 轻度：面积 < 5%，零星分布
    MODERATE = "moderate"    # 中度：面积 5-15%，或条带状分布
    SEVERE = "severe"        # 重度：面积 15-30%，成片分布
    CRITICAL = "critical"    # 极重：面积 > 30%，大面积连片


class UrgencyLevel(Enum):
    """处理紧急程度"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class DistributionPattern(Enum):
    """异常空间分布模式"""
    NONE = "none"                # 无异常
    SCATTERED = "scattered"      # 零星分散
    STRIP = "strip"              # 条带状（沿播种行方向）
    CLUSTERED = "clustered"      # 成片聚集
    WIDESPREAD = "widespread"    # 大面积弥漫


@dataclass
class AnomalyRegion:
    """单个异常区域"""
    bbox: BBox                                # 区域边界框
    area_pixels: int = 0                      # 区域面积（像素）
    area_ratio: float = 0.0                   # 面积占比 (0-1)
    position_desc: str = ""                   # 位置描述（如"东北角", "中部偏左"）
    anomaly_type: str = ""                    # 异常类型 (double_plant / planter_skip)
    confidence: float = 0.0                   # 该区域的置信度


@dataclass
class AnomalyDetail:
    """单类异常的详细分析"""
    label: str                                     # 异常类别名 (e.g. "double_plant")
    detected: bool = False                         # 是否检测到
    confidence: float = 0.0                        # 置信度 (0-1)
    affected_area_ratio: float = 0.0               # 受影响面积占比 (0-1)
    severity: SeverityLevel = SeverityLevel.NONE
    distribution: DistributionPattern = DistributionPattern.NONE
    regions: List[AnomalyRegion] = field(default_factory=list)
    visual_evidence: str = ""                      # VLM给出的视觉现象描述
    evidence_refs: List[str] = field(default_factory=list)  # 引用的证据ID [E1, E3, ...]


@dataclass
class SpatialAnalysis:
    """空间分布摘要"""
    total_anomaly_regions: int = 0            # 异常区域总数
    total_affected_ratio: float = 0.0         # 总受影响面积比
    primary_location: str = ""                # 主要异常位置描述
    distribution_pattern: DistributionPattern = DistributionPattern.NONE
    pattern_description: str = ""             # 分布模式的自然语言描述
    regions: List[AnomalyRegion] = field(default_factory=list)


@dataclass
class Recommendation:
    """单条补救建议"""
    action: str                               # 建议操作
    urgency: UrgencyLevel = UrgencyLevel.LOW
    target_anomaly: str = ""                  # 针对的异常类型
    target_area: str = ""                     # 目标区域描述
    expected_effect: str = ""                 # 预期效果
    notes: str = ""                           # 附加说明

@dataclass
class SingleAnomalVLMResult:
    """
    VLM 对单类播种异常（double_plant 或 planter_skip）的专项分析结果。
    设计偏叙述性，主要供 Phase 3 报告渲染使用。
    """
    

@dataclass
class AgentReport:
    """Agent 分析报告（完整输出）"""
    agent_name: str                           # Agent名称
    image_path: str = ""                      # 分析的图像路径
    tile_id: str = ""                         # 图像对应的tile_id

    # 核心分析结果
    detected_anomalies: List[AnomalyDetail] = field(default_factory=list)
    overall_severity: SeverityLevel = SeverityLevel.NONE
    spatial_summary: Optional[SpatialAnalysis] = None
    recommendations: List[Recommendation] = field(default_factory=list)

    # 置信度与证据
    overall_confidence: float = 0.0
    evidence_refs: List[str] = field(default_factory=list)

    # VLM 原始输出（可追溯）
    raw_vlm_output: str = ""
    raw_general_prediction: Optional[Dict[str, Any]] = None

    # 元信息
    summary: str = ""                         
    urgency: UrgencyLevel = UrgencyLevel.LOW

    def to_dict(self) -> Dict[str, Any]:
        """序列化为字典（方便JSON输出）"""
        return {
            "agent_name": self.agent_name,
            "image_path": self.image_path,
            "tile_id": self.tile_id,
            "summary": self.summary,
            "overall_severity": self.overall_severity.value,
            "overall_confidence": self.overall_confidence,
            "urgency": self.urgency.value,
            "detected_anomalies": [
                {
                    "label": a.label,
                    "detected": a.detected,
                    "confidence": a.confidence,
                    "affected_area_ratio": a.affected_area_ratio,
                    "severity": a.severity.value,
                    "distribution": a.distribution.value,
                    "visual_evidence": a.visual_evidence,
                    "evidence_refs": a.evidence_refs,
                    "regions": [
                        {
                            "bbox": list(r.bbox),
                            "area_ratio": r.area_ratio,
                            "position_desc": r.position_desc,
                            "confidence": r.confidence,
                        }
                        for r in a.regions
                    ],
                }
                for a in self.detected_anomalies
            ],
            "spatial_summary": {
                "total_anomaly_regions": self.spatial_summary.total_anomaly_regions,
                "total_affected_ratio": self.spatial_summary.total_affected_ratio,
                "primary_location": self.spatial_summary.primary_location,
                "distribution_pattern": self.spatial_summary.distribution_pattern.value,
                "pattern_description": self.spatial_summary.pattern_description,
            } if self.spatial_summary else None,
            "recommendations": [
                {
                    "action": r.action,
                    "urgency": r.urgency.value,
                    "target_anomaly": r.target_anomaly,
                    "target_area": r.target_area,
                    "expected_effect": r.expected_effect,
                    "notes": r.notes,
                }
                for r in self.recommendations
            ],
            "evidence_refs": self.evidence_refs,
            "raw_vlm_output": self.raw_vlm_output,
        }