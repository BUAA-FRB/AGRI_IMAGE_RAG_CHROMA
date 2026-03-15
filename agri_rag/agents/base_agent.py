"""
base_agent.py — 异常检测专项Agent基类

所有专项Agent（播种异常、水害、营养缺乏等）都继承此基类，
确保统一接口，便于 AgentRouter 调度。
"""

from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from ..types import AgentReport, RetrievalHit


class BaseAnomalyAgent(ABC):
    """
    专项异常分析Agent基类

    子类需要实现：
    - HANDLED_LABELS: 该Agent负责处理的标签列表
    - analyze(): 核心分析方法
    """

    # 子类覆盖：该Agent能处理的标签类别
    HANDLED_LABELS: List[str] = []

    # Agent名称（子类覆盖）
    AGENT_NAME: str = "base_agent"

    def should_activate(self, general_prediction: Dict[str, Any]) -> bool:
        """
        根据通用RAG预测结果判断是否需要激活该Agent。

        判断逻辑：
        1. 从通用预测的JSON输出中提取已检测到的标签
        2. 与 HANDLED_LABELS 取交集
        3. 有交集则激活

        Args:
            general_prediction: 通用RAG流程的JSON输出
                支持多种格式：
                - {"conclusion": "...", "evidence": [{"label": "double_plant", ...}]}
                - {"labels": ["double_plant", "planter_skip"]}
                - 原始文本字符串
        """
        detected = self._extract_labels_from_prediction(general_prediction)
        return any(lb in self.HANDLED_LABELS for lb in detected)

    @abstractmethod
    def analyze(
        self,
        image_path: str,
        general_prediction: Dict[str, Any],
        hits: List[RetrievalHit],
        patch_hits: Optional[List[RetrievalHit]] = None,
        qwen_client: Optional[Any] = None,
    ) -> AgentReport:
        """
        执行专项分析。

        Args:
            image_path: 待分析图像路径
            general_prediction: 通用RAG流程输出
            hits: 全局级检索结果
            patch_hits: patch级检索结果（可选，用于空间定位）
            qwen_client: Qwen3VLClient 实例（用于二次推理，Phase 2使用）

        Returns:
            AgentReport: 结构化分析报告
        """
        ...

    # ================================================================
    # 工具方法
    # ================================================================

    @staticmethod
    def _extract_labels_from_prediction(prediction: Dict[str, Any]) -> List[str]:
        """
        从通用预测结果中提取检测到的标签列表。
        兼容多种输出格式。
        """
        labels: List[str] = []

        if not prediction:
            return labels

        # 格式1: {"evidence": [{"label": "xxx", ...}, ...]}
        evidence_list = prediction.get("evidence", [])
        if isinstance(evidence_list, list):
            for item in evidence_list:
                if isinstance(item, dict):
                    lb = item.get("label", "")
                    if lb:
                        labels.append(str(lb).strip())

        # 格式2: {"labels": ["xxx", ...]}
        labels_field = prediction.get("labels", [])
        if isinstance(labels_field, list):
            labels.extend([str(lb).strip() for lb in labels_field if lb])

        # 格式3: {"conclusion": "检测到 double_plant 和 planter_skip"}
        conclusion = prediction.get("conclusion", "")
        if isinstance(conclusion, str) and conclusion:
            # 用已知标签名做关键词匹配
            known_labels = [
                "double_plant", "planter_skip", "drydown", "endrow",
                "nutrient_deficiency", "storm_damage", "water",
                "waterway", "weed_cluster",
            ]
            for kl in known_labels:
                if kl in conclusion.lower().replace(" ", "_"):
                    labels.append(kl)

        # 去重，保持顺序
        seen = set()
        unique = []
        for lb in labels:
            if lb not in seen:
                seen.add(lb)
                unique.append(lb)
        return unique

    @staticmethod
    def _parse_labels_from_metadata(hit: RetrievalHit) -> List[str]:
        """从检索结果的metadata中解析labels_present字段"""
        m = hit.metadata or {}
        raw = m.get("labels_present", "")
        if not raw:
            return []
        # labels_present 在 Chroma 中存为 JSON string
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, list):
                return [str(x) for x in parsed]
        except (json.JSONDecodeError, TypeError):
            pass
        # 兼容非JSON格式
        if isinstance(raw, str):
            return [x.strip() for x in raw.split(",") if x.strip()]
        return []

    @staticmethod
    def _parse_bbox_from_metadata(hit: RetrievalHit) -> Optional[tuple]:
        """从metadata中解析bbox字段（存为字符串格式）"""
        m = hit.metadata or {}
        raw = m.get("bbox", "")
        if not raw:
            return None
        try:
            # "(x1, y1, x2, y2)" -> tuple
            nums = re.findall(r"\d+", str(raw))
            if len(nums) == 4:
                return tuple(int(n) for n in nums)
        except Exception:
            pass
        return None

    @staticmethod
    def _filter_hits_by_labels(
        hits: List[RetrievalHit],
        target_labels: List[str],
        distance_threshold: float = 1.5,
    ) -> List[RetrievalHit]:
        """
        过滤检索结果，只保留包含目标标签的hits。

        Args:
            hits: 检索结果列表
            target_labels: 目标标签列表
            distance_threshold: 距离阈值（过滤掉太远的结果）

        Returns:
            过滤后的hits（保持原始顺序）
        """
        filtered = []
        for h in hits:
            if h.distance > distance_threshold:
                continue
            labels = BaseAnomalyAgent._parse_labels_from_metadata(h)
            if any(lb in target_labels for lb in labels):
                filtered.append(h)
        return filtered