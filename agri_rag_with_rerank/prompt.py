from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from .types import RetrievalHit


def build_prompt(user_query: str, hits: List[RetrievalHit], max_hits: int = 8) -> str:
    lines = []
    lines.append("你是农业遥感异常/灾害识别助手。请基于下面检索到的证据回答用户问题。")
    lines.append("要求：给出结论，并引用证据（用 [E1],[E2]... 标注），不要编造未在证据中出现的信息。")
    lines.append("")
    lines.append(f"用户问题：{user_query}")
    lines.append("")
    lines.append("检索证据（按相似度从高到低，distance 越小越相似）：")

    for i, h in enumerate(hits[:max_hits], 1):
        m = h.metadata or {}
        labels = m.get("labels_present", None)
        bbox = m.get("bbox", None)
        split = m.get("split", None)
        tile_id = m.get("tile_id", None)
        path = m.get("path", None)
        nir_path = m.get("nir_path", None)

        lines.append(f"[E{i}] distance={h.distance:.4f}")
        lines.append(f"     tile_id={tile_id} split={split}")
        lines.append(f"     rgb_path={path}")
        if nir_path:
            lines.append(f"     nir_path={nir_path}")
        if bbox:
            lines.append(f"     bbox={bbox}")
        if labels:
            lines.append(f"     labels_present={labels}")

    lines.append("")
    lines.append("请输出：")
    lines.append("1) 结论（尽量具体到灾害/异常类别）")
    lines.append("2) 依据（逐条引用证据 [E#]，说明哪些现象支持你的判断）")
    lines.append("3) 不确定性（如果证据不足，说明还需要哪些信息/更合适的证据）")
    return "\n".join(lines)


def _to_local_path(p: str) -> str:
    return str(Path(p).resolve())


def _json_schema_text() -> str:
    return (
        "\n\n【输出格式约束】\n"
        "请只输出一个 JSON 对象（不要输出其它多余文字）。\n"
        "JSON schema 示例：\n"
        "{\n"
        '  "predicted_labels": [\n'
        '    {"label":"类别A","confidence":0.0,"refs":["E1","E3"],"reason":"..."}\n'
        "  ],\n"
        '  "conclusion": "一句话总结（可选）",\n'
        '  "uncertainty": "不确定性与还需要的证据（可选）"\n'
        "}\n"
        "\n"
        "说明：\n"
        "- label 必须是灾害/异常类别名称（可多类）。\n"
        "- confidence ∈ [0,1]，refs 用于引用证据编号（如 E1、E2）。\n"
    )


def build_qwen3vl_messages(
    user_query: str,
    hits: List[RetrievalHit],
    query_image_path: Optional[str] = None,
    max_hits: int = 6,
    include_nir_evidence: bool = False,
    output_json: bool = False,
) -> List[Dict[str, Any]]:
    prompt_text = build_prompt(user_query, hits, max_hits=max_hits)
    if output_json:
        prompt_text += _json_schema_text()

    content: List[Dict[str, Any]] = []

    if query_image_path:
        content.append({"type": "text", "text": "【待预测图像】如下图是需要进行灾害/异常识别的目标图像："})
        content.append({"type": "image", "image": _to_local_path(query_image_path)})

    content.append({"type": "text", "text": "【RAG检索证据图像】下面依次给出检索到的证据图片，顺序与 [E1],[E2]... 一一对应："})
    for i, h in enumerate(hits[:max_hits], 1):
        m = h.metadata or {}
        rgb_path = m.get("path", "")
        nir_path = m.get("nir_path", "") if include_nir_evidence else ""

        meta_line = f"[E{i}] distance={h.distance:.4f} tile_id={m.get('tile_id','')} split={m.get('split','')}"
        content.append({"type": "text", "text": meta_line})

        if rgb_path:
            content.append({"type": "image", "image": _to_local_path(str(rgb_path))})
        if nir_path:
            content.append({"type": "text", "text": f"[E{i}] NIR（可选证据）"})
            content.append({"type": "image", "image": _to_local_path(str(nir_path))})

    content.append({"type": "text", "text": prompt_text})
    return [{"role": "user", "content": content}]


def _parse_labels_present(labels_present: str) -> List[str]:
    """
    labels_present 在库里经常是 JSON 字符串：["endrow","double_plant"]
    这里尽量解析成 label 列表，解析失败则返回原字符串（简化处理）。
    """
    if not labels_present or not isinstance(labels_present, str):
        return []
    s = labels_present.strip()
    try:
        obj = json.loads(s)
        if isinstance(obj, list):
            out = []
            for it in obj:
                if isinstance(it, str) and it.strip():
                    out.append(it.strip())
            return out
    except Exception:
        pass
    # 兜底：拆分
    for sep in [",", "，", "、", "|", ";"]:
        if sep in s:
            parts = [p.strip().strip('"').strip("'") for p in s.split(sep)]
            parts = [p for p in parts if p]
            return parts
    return [s]


def build_qwen_rerank_messages(
    user_query: str,
    hits: List[RetrievalHit],
    max_hits: int = 30,
) -> List[Dict[str, Any]]:
    """
    rerank prompt（关键改动）：
      - 不再给 distance 数值，避免模型把 distance 当成 score “照抄”
      - 只给 sim_rank（候选在 stage-1 的相似度排序位置），可做 tie-break，但不是 score
      - 给出“可用 labels 集合”，强制模型 grounding 在候选元信息上进行语义判断
      - 强约束：必须输出恰好 k 条 ranking 覆盖全部 eid
    """
    cand = hits[:max_hits]
    k = len(cand)
    eids = [f"E{i}" for i in range(1, k + 1)]

    # 汇总候选里出现过的 labels，给模型一个“可用概念集合”
    label_set = set()
    cand_labels: List[List[str]] = []
    for h in cand:
        m = h.metadata or {}
        lbs = _parse_labels_present(str(m.get("labels_present", "") or ""))
        cand_labels.append(lbs)
        for lb in lbs:
            label_set.add(lb)

    labels_all = sorted(label_set)

    lines: List[str] = []
    lines.append("你是检索重排序（rerank）助手。你的输出会被程序自动解析。")
    lines.append("")
    lines.append("【任务】为每个候选证据 Ei 评估其与用户问题的语义相关性，输出 score ∈ [0,1]，并按 score 从高到低排序。")
    lines.append("")
    lines.append("【非常重要：语义相关性定义】")
    lines.append("score 表示“候选的 labels_present / tile_id / 其它元信息”与用户问题的语义匹配程度。")
    lines.append("例如：用户问题提到某个灾害/异常类别、关键词或同义表达，而候选 labels_present 中包含该类别或高度相关类别，则 score 应更高。")
    lines.append("")
    lines.append("【硬性要求（必须满足）】")
    lines.append("1) 只输出一个 JSON 对象；不要输出任何额外文字；不要 Markdown 代码块。")
    lines.append(f"2) 你必须输出恰好 k={k} 条 ranking，且必须覆盖全部 eid：{', '.join(eids)}。")
    lines.append("3) 每个 eid 必须出现且只能出现一次；不允许漏项；不允许 score 为 null。")
    lines.append("4) score 不允许直接等于 sim_rank 或其它输入数值；sim_rank 只能用于并列时 tie-break。")
    lines.append("5) 如果你无法从元信息判断语义，请让 score 接近并列（例如都在 0.45~0.55），再用 sim_rank 进行轻微区分。")
    lines.append("")
    lines.append(f"用户问题：{user_query}")
    lines.append("")
    if labels_all:
        lines.append("候选中出现过的可用 labels 集合（仅供你理解候选语义，不要编造集合外的新标签）：")
        lines.append(", ".join(labels_all))
        lines.append("")

    lines.append("候选证据（sim_rank 越小表示 stage-1 更相似，仅可做 tie-break）：")
    for i, h in enumerate(cand, 1):
        m = h.metadata or {}
        tile_id = str(m.get("tile_id", "") or "")
        split = str(m.get("split", "") or "")
        lbs = cand_labels[i - 1]
        lbs_text = "|".join(lbs) if lbs else ""
        # 关键：不给 distance 数值
        lines.append(f"- E{i}: sim_rank={i}/{k}, tile_id={tile_id}, split={split}, labels_present={lbs_text}")

    lines.append("")
    lines.append("【输出 JSON schema】")
    lines.append('{ "ranking": [ {"eid":"E1","score":0.0,"reason":"..."} ] }')
    lines.append(f"再次强调：ranking 必须包含 {k} 条，且必须包含全部 eid：{', '.join(eids)}。")
    lines.append("reason 请简短说明你为什么给这个 score（引用 labels_present 或问题关键词即可）。")

    content = [{"type": "text", "text": "\n".join(lines)}]
    return [{"role": "user", "content": content}]
