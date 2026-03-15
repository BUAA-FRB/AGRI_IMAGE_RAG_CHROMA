"""
planting_prompts.py — 播种异常专项 Prompt 模板

为播种异常Agent的Qwen2-VL二次推理提供专业化的prompt，
区别于通用RAG prompt，聚焦于播种行规律性、行间距、植株密度等视觉特征。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from ..types import RetrievalHit, SpatialAnalysis


def build_planting_analysis_prompt(
    pre_analysis: Dict[str, Any],
    spatial_info: Optional[SpatialAnalysis] = None,
) -> str:
    """
    构建播种异常的专项分析 prompt（纯文本部分）。

    Args:
        pre_analysis: 阶段1-3的预分析结果，包含：
            - filtered_hits: 播种相关的证据摘要
            - severity_estimate: 初步严重度
            - area_ratio: 初步面积占比
        spatial_info: 空间分析结果（如果有patch级数据）
    """
    lines = []

    lines.append("你是一位专业的精准农业播种异常分析专家。")
    lines.append("你需要基于以下预分析信息和证据图像，对农田图像进行播种异常的深度分析。")
    lines.append("")
    lines.append("【分析重点】")
    lines.append("请特别关注以下播种相关的视觉特征：")
    lines.append("  - 播种行的规律性：行间距是否均匀，是否有异常密集或缺失的行")
    lines.append("  - 植株密度变化：同一行内植株是否过密（重播）或出现明显间隙（漏播）")
    lines.append("  - 行间空隙模式：是否有整行或连续多行缺失的情况")
    lines.append("  - 颜色/长势差异：重播区域可能因竞争导致植株矮小、黄化")
    lines.append("")

    # 注入预分析上下文
    lines.append("【预分析结果】")
    severity = pre_analysis.get("severity_estimate", "未知")
    area_ratio = pre_analysis.get("area_ratio", 0.0)
    detected_types = pre_analysis.get("detected_types", [])
    lines.append(f"  初步检测到的异常类型：{', '.join(detected_types) if detected_types else '待确认'}")
    lines.append(f"  初步严重度估计：{severity}")
    lines.append(f"  初步受影响面积比：{area_ratio:.1%}" if isinstance(area_ratio, float) else f"  初步受影响面积比：{area_ratio}")

    if spatial_info:
        lines.append(f"  异常区域数量：{spatial_info.total_anomaly_regions}")
        lines.append(f"  主要位置：{spatial_info.primary_location}")
        lines.append(f"  分布模式：{spatial_info.distribution_pattern.value}")

    lines.append("")

    # 证据摘要
    evidence_summary = pre_analysis.get("evidence_summary", [])
    if evidence_summary:
        lines.append("【播种相关检索证据摘要】")
        for i, ev in enumerate(evidence_summary, 1):
            lines.append(f"  [E{ev.get('original_index', i)}] "
                         f"distance={ev.get('distance', 0):.4f} "
                         f"labels={ev.get('labels', [])} "
                         f"tile_id={ev.get('tile_id', '')}")
        lines.append("")

    # 输出要求
    lines.append("【输出格式约束】")
    lines.append("请只输出一个 JSON 对象（不要输出其它多余文字）。")
    lines.append("JSON schema：")
    lines.append("""{
  "planting_analysis": {
    "double_plant": {
      "detected": true/false,
      "confidence": 0.0,
      "affected_rows_desc": "受影响播种行的描述",
      "visual_evidence": "具体视觉现象描述（行内密度、颜色、长势等）"
    },
    "planter_skip": {
      "detected": true/false,
      "confidence": 0.0,
      "gap_pattern": "空行/断行的模式描述",
      "visual_evidence": "具体视觉现象描述（缺失行数、间隔、连续性等）"
    }
  },
  "spatial_description": "异常区域的空间分布自然语言描述",
  "growth_stage_estimate": "估计的作物生长阶段（如V2-V4、VT-R1等）",
  "overall_severity": "none/minor/moderate/severe/critical",
  "evidence_reasoning": "基于 [E#] 证据的推理过程"
}""")

    return "\n".join(lines)


def build_planting_vlm_messages(
    image_path: str,
    filtered_hits: List[RetrievalHit],
    pre_analysis: Dict[str, Any],
    spatial_info: Optional[SpatialAnalysis] = None,
    max_evidence: int = 4,
) -> List[Dict[str, Any]]:
    """
    构建播种专项的多模态VLM消息。

    与通用 build_qwen3vl_messages 的区别：
    1. prompt 聚焦播种特征而非通用灾害
    2. 只附带播种相关的证据图像
    3. 注入了预分析上下文（空间信息、严重度估计）
    4. 输出schema针对播种场景定制
    """
    prompt_text = build_planting_analysis_prompt(pre_analysis, spatial_info)

    content: List[Dict[str, Any]] = []

    # 1) 待分析图像
    content.append({
        "type": "text",
        "text": "【待分析农田图像】请仔细观察以下农田图像中的播种行模式和植株分布："
    })
    content.append({
        "type": "image",
        "image": str(Path(image_path).resolve())
    })

    # 2) 播种相关证据图像（仅限筛选后的）
    if filtered_hits:
        content.append({
            "type": "text",
            "text": "【播种异常参考证据】以下是检索到的与播种异常最相关的参考图像："
        })
        for i, h in enumerate(filtered_hits[:max_evidence], 1):
            m = h.metadata or {}
            rgb_path = m.get("path", "")
            meta_line = (
                f"[E{i}] distance={h.distance:.4f} "
                f"tile_id={m.get('tile_id', '')} "
                f"labels={m.get('labels_present', '')}"
            )
            content.append({"type": "text", "text": meta_line})
            if rgb_path:
                content.append({
                    "type": "image",
                    "image": str(Path(rgb_path).resolve())
                })

    # 3) 专项分析prompt
    content.append({"type": "text", "text": prompt_text})

    return [{"role": "user", "content": content}]