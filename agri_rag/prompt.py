from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from .types import RetrievalHit


def build_prompt(user_query: str, hits: List[RetrievalHit], max_hits: int = 8) -> str:
    """
    Build a RAG prompt (text-only) that references retrieved image evidence.
    You can feed the returned string to any LLM.

    Tip:
    - If you use a multimodal LLM, you can additionally attach the actual images referred by `path`.
    """
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


def _to_file_uri(p: str) -> str:
    # transformers/qwen processor expects a local file path, not file:// URI
    return str(Path(p).resolve())


def build_qwen3vl_messages(
    user_query: str,
    hits: List[RetrievalHit],
    query_image_path: Optional[str] = None,
    max_hits: int = 6,
    include_nir_evidence: bool = False,
    output_json: bool = False,
) -> List[Dict[str, Any]]:
    """
    Build OpenAI-style multimodal messages for Qwen3-VL (Transformers apply_chat_template).
    Key idea: attach evidence images in the same order as [E1],[E2],... to align with build_prompt().

    - query_image_path: if provided, it will be placed at the beginning as "the image to be predicted".
    - include_nir_evidence: optionally attach NIR images for each evidence if available.
    - output_json: ask model to output strict JSON (useful for downstream parsing).
    """
    prompt_text = build_prompt(user_query, hits, max_hits=max_hits)

    if output_json:
        prompt_text += (
            "\n\n【输出格式约束】\n"
            "请只输出一个 JSON 对象（不要输出其它多余文字）。\n"
            "JSON schema 示例：\n"
            "{\n"
            '  "conclusion": "灾害/异常类别（可多类）",\n'
            '  "evidence": [\n'
            '    {"label":"类别A","confidence":0.0,"refs":["E1","E3"],"reason":"..."}\n'
            "  ],\n"
            '  "uncertainty": "不确定性与还需要的证据"\n'
            "}\n"
        )

    content: List[Dict[str, Any]] = []

    # 1) Optional query image (the image we want to predict)
    if query_image_path:
        content.append({"type": "text", "text": "【待预测图像】如下图是需要进行灾害/异常识别的目标图像："})
        content.append({"type": "image", "image": _to_file_uri(query_image_path)})

    # 2) Evidence images in order E1..Ek
    content.append({"type": "text", "text": "【RAG检索证据图像】下面依次给出检索到的证据图片，顺序与 [E1],[E2]... 一一对应："})
    for i, h in enumerate(hits[:max_hits], 1):
        m = h.metadata or {}
        rgb_path = m.get("path", "")
        nir_path = m.get("nir_path", "") if include_nir_evidence else ""

        # Tag + metadata (keep it short; full metadata is in prompt_text)
        meta_line = f"[E{i}] distance={h.distance:.4f} tile_id={m.get('tile_id','')} split={m.get('split','')}"
        content.append({"type": "text", "text": meta_line})

        if rgb_path:
            content.append({"type": "image", "image": _to_file_uri(str(rgb_path))})
        if nir_path:
            # NIR may be grayscale; processor will handle as image
            content.append({"type": "text", "text": f"[E{i}] NIR（可选证据）"})
            content.append({"type": "image", "image": _to_file_uri(str(nir_path))})

    # 3) Text prompt (the structured evidence list + output requirements)
    content.append({"type": "text", "text": prompt_text})

    return [{"role": "user", "content": content}]
