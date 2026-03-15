"""
agent_router.py — Agent 路由分发器

根据通用RAG预测结果，自动分发到对应的专项Agent执行分析。
支持多Agent并行激活（一张图可能同时存在多种异常）。
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from .base_agent import BaseAnomalyAgent
from .planting_agent import PlantingAnomalyAgent
from ..types import AgentReport, RetrievalHit

logger = logging.getLogger(__name__)


class AgentRouter:
    """
    Agent 路由分发器

    使用方式：
        router = AgentRouter()
        reports = router.route(
            image_path="xxx.jpg",
            general_prediction={"conclusion": "...", "evidence": [...]},
            hits=hits,
        )
        for report in reports:
            print(report.to_dict())
    """

    def __init__(self, agents: Optional[List[BaseAnomalyAgent]] = None):
        """
        Args:
            agents: 自定义Agent列表。默认注册所有内置Agent。
        """
        if agents is not None:
            self.agents = agents
        else:
            # 默认注册内置Agent
            self.agents: List[BaseAnomalyAgent] = [
                PlantingAnomalyAgent(),
                # 后续扩展：
                # WaterDamageAgent(),
                # NutrientDeficiencyAgent(),
                # StormDamageAgent(),
            ]
        logger.info(f"[AgentRouter] 已注册 {len(self.agents)} 个专项Agent: "
                     f"{[a.AGENT_NAME for a in self.agents]}")

    def route(
        self,
        image_path: str,
        general_prediction: Dict[str, Any],
        hits: List[RetrievalHit],
        patch_hits: Optional[List[RetrievalHit]] = None,
        qwen_client: Optional[Any] = None,
        force_agents: Optional[List[str]] = None,
    ) -> List[AgentReport]:
        """
        路由分发：遍历所有Agent，激活匹配的Agent执行分析。

        Args:
            image_path: 待分析图像路径
            general_prediction: 通用RAG流程输出
            hits: 全局检索结果
            patch_hits: patch级检索结果
            qwen_client: Qwen3VLClient实例
            force_agents: 强制激活的Agent名称列表（跳过should_activate检查）

        Returns:
            所有激活Agent的分析报告列表
        """
        reports: List[AgentReport] = []

        for agent in self.agents:
            # 检查是否需要激活
            should_run = False
            if force_agents and agent.AGENT_NAME in force_agents:
                should_run = True
                logger.info(f"[AgentRouter] 强制激活 {agent.AGENT_NAME}")
            elif agent.should_activate(general_prediction):
                should_run = True
                logger.info(f"[AgentRouter] 自动激活 {agent.AGENT_NAME}")
            else:
                logger.debug(f"[AgentRouter] 跳过 {agent.AGENT_NAME}（未检测到相关异常）")

            if not should_run:
                continue

            try:
                report = agent.analyze(
                    image_path=image_path,
                    general_prediction=general_prediction,
                    hits=hits,
                    patch_hits=patch_hits,
                    qwen_client=qwen_client,
                )
                reports.append(report)
                logger.info(
                    f"[AgentRouter] {agent.AGENT_NAME} 分析完成: "
                    f"severity={report.overall_severity.value}, "
                    f"confidence={report.overall_confidence:.2f}"
                )
            except Exception as e:
                logger.error(f"[AgentRouter] {agent.AGENT_NAME} 执行失败: {e}", exc_info=True)

        if not reports:
            logger.info("[AgentRouter] 无Agent被激活，所有结果为通用RAG输出")

        return reports

    def get_agent(self, name: str) -> Optional[BaseAnomalyAgent]:
        """根据名称获取Agent实例"""
        for agent in self.agents:
            if agent.AGENT_NAME == name:
                return agent
        return None

    def list_agents(self) -> List[Dict[str, Any]]:
        """列出所有已注册的Agent"""
        return [
            {
                "name": agent.AGENT_NAME,
                "handled_labels": agent.HANDLED_LABELS,
                "class": agent.__class__.__name__,
            }
            for agent in self.agents
        ]