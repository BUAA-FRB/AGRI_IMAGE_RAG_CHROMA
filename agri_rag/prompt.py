# agri_rag/prompt.py
from __future__ import annotations

import ast
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image

from .types import RetrievalHit
from .image_utils import open_rgb, open_gray, nir_to_rgb


# =========================
# Prompt (compact but professional)
# =========================
def build_prompt(
    user_query: str,
    hits: List[RetrievalHit],
    max_hits: int = 8,
    include_evidence: bool = True,
) -> str:
    """
    目标：在“专业性/可控性”与“显存友好（token 更短）”之间取平衡。
    - 默认 include_evidence=True：用于纯文本 prompt（没有证据图时，需要把证据以文本形式给 LLM）。
    - 在 build_qwen3vl_messages()（多模态、有证据图）里会传 include_evidence=False，避免重复叠加 token。
    """
    lines: List[str] = []

    # 角色 + 规则（短、硬、可执行）
    lines.append("你是农业遥感异常/灾害判读助手（RAG 证据约束）。")
    lines.append("规则：只能依据证据 [E1..]（以及待预测图像）推断；禁止编造证据外信息；结论/理由必须引用 [E#]。")
    lines.append("判读写法：现象(色调/纹理/形态/边界/连通) → 解释(成因/过程) → 区分点(与相近类别) → 结论 → 建议。")
    lines.append("")
    lines.append(f"用户问题：{user_query}".strip())
    lines.append("")

    # 证据文本（纯文本模式需要；多模态模式不需要重复列一遍）
    if include_evidence:
        lines.append("证据元信息（与 [E1],[E2]... 对应；distance 越小越相似）：")
        for i, h in enumerate(hits[:max_hits], 1):
            m = h.metadata or {}
            typ = str(m.get("type", "") or "")
            tile_id = str(m.get("tile_id", "") or "")
            split = str(m.get("split", "") or "")
            bbox = m.get("bbox", None)
            labels = m.get("labels_present", None)

            dist = getattr(h, "distance", None)
            try:
                dist_str = f"{float(dist):.4f}" if dist is not None else "NA"
            except Exception:
                dist_str = "NA"

            parts = [f"[E{i}]", f"type={typ}", f"dist={dist_str}"]
            if tile_id:
                parts.append(f"tile_id={tile_id}")
            if split:
                parts.append(f"split={split}")
            if bbox:
                parts.append(f"bbox={bbox}")
            if labels:
                # labels_present 通常是 JSON 字符串（保持原样，避免解析失败产生额外 token）
                parts.append(f"labels_present={labels}")

            lines.append(" - " + " ".join(parts))

        lines.append("")

    # 输出要求（JSON/非 JSON 的约束由 _json_schema_text 追加）
    lines.append("请输出：")
    lines.append("1) conclusion：专业判读结论（建议 4–8 句，覆盖：类别 + 关键特征 + 空间形态/范围 + 区分点 + 建议；句末可加 [E#]）。")
    lines.append("2) predicted_labels：给出类别(label)与置信度(confidence 0~1)及 refs(如 [\"E1\",\"E3\"])。")
    lines.append("3) uncertainty：若证据不足或存在混淆，说明不确定点与需要补充的信息（如更高分辨率/多光谱/时序对比/地面核验）。")

    return "\n".join(lines)


def _json_schema_text() -> str:
    """
    重点：严格可解析，但不要写太长（避免 token 爆炸）。
    """
    return (
        "\n\n"
        "【输出格式：严格 JSON】\n"
        "你必须只输出一个 JSON 对象（不要输出任何额外文字；不要 Markdown；不要 ```json 代码块）。\n"
        "字段要求：\n"
        "- predicted_labels: list[ {label:str, confidence:float, refs:list[str], reason:str} ]\n"
        "- conclusion: str（建议 4–8 句；必须引用证据编号，如 [E2]；更专业、更具体）\n"
        "- uncertainty: str（可选；证据不足必须给）\n"
        "硬约束：输出必须以 '{' 开始，以 '}' 结束；使用英文双引号；不允许尾随逗号。\n"
    )


# =========================
# Image utilities (resize to reduce visual tokens)
# =========================
def _to_local_path(p: str) -> str:
    return str(Path(p).resolve())


def _vl_max_side() -> int:
    """
    不减少图像数量，但通过缩小每张图的最大边长来降低视觉 token 和显存。
    可用环境变量覆盖：
      AGRI_RAG_VL_MAX_SIDE=448 / 384 / 320 ...
    """
    v = os.getenv("AGRI_RAG_VL_MAX_SIDE", "512").strip()
    try:
        n = int(v)
        return max(224, n)  # 下限给到 224，避免太糊
    except Exception:
        return 512


def _image_cache_dir() -> str:
    return os.path.abspath(os.getenv("AGRI_RAG_VL_IMAGE_CACHE_DIR", "./logs/vl_image_cache"))


def _cache_path_for_image(src_path: str, max_side: int, kind: str) -> str:
    key = f"{os.path.abspath(src_path)}|max_side={max_side}|{kind}"
    hid = hashlib.md5(key.encode("utf-8")).hexdigest()
    d = _image_cache_dir()
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, f"{hid}_{kind}_{max_side}.png")


def _save_resized_rgb(img: Image.Image, outp: str, max_side: int) -> str:
    # PIL thumbnail 会保持长宽比，并保证不超过 max_side
    im = img.convert("RGB")
    im.thumbnail((max_side, max_side), Image.BICUBIC)
    im.save(outp)
    return outp


def _ensure_resized_rgb_image(path: str, max_side: int, kind: str) -> str:
    outp = _cache_path_for_image(path, max_side, kind)
    if os.path.exists(outp):
        return outp
    img = open_rgb(path)
    return _save_resized_rgb(img, outp, max_side)


def _ensure_resized_nir_rgb_image(nir_path: str, max_side: int, kind: str) -> str:
    outp = _cache_path_for_image(nir_path, max_side, kind)
    if os.path.exists(outp):
        return outp
    nir = open_gray(nir_path)
    nir_rgb = nir_to_rgb(nir)  # -> RGB
    return _save_resized_rgb(nir_rgb, outp, max_side)


# =========================
# BBox / patch crop utilities
# =========================
def _parse_bbox_any(bbox: Any) -> Optional[Tuple[int, int, int, int]]:
    if bbox is None:
        return None
    if isinstance(bbox, (tuple, list)) and len(bbox) == 4:
        try:
            x0, y0, x1, y1 = [int(float(x)) for x in bbox]
            return (x0, y0, x1, y1)
        except Exception:
            return None
    if isinstance(bbox, str) and bbox.strip():
        s = bbox.strip()
        try:
            obj = ast.literal_eval(s)
            if isinstance(obj, (tuple, list)) and len(obj) == 4:
                x0, y0, x1, y1 = [int(float(x)) for x in obj]
                return (x0, y0, x1, y1)
        except Exception:
            pass
    return None


def _clamp_bbox(b: Tuple[int, int, int, int], w: int, h: int) -> Tuple[int, int, int, int]:
    x0, y0, x1, y1 = b
    x0 = max(0, min(x0, w))
    x1 = max(0, min(x1, w))
    y0 = max(0, min(y0, h))
    y1 = max(0, min(y1, h))
    if x1 <= x0:
        x1 = min(w, x0 + 1)
    if y1 <= y0:
        y1 = min(h, y0 + 1)
    return (x0, y0, x1, y1)


def _patch_cache_dir() -> str:
    return os.path.abspath(os.getenv("AGRI_RAG_PATCH_CACHE_DIR", "./logs/patch_cache"))


def _cache_path_for_patch(src_path: str, bbox: Tuple[int, int, int, int], kind: str, max_side: int) -> str:
    key = f"{os.path.abspath(src_path)}|bbox={bbox}|{kind}|max_side={max_side}"
    hid = hashlib.md5(key.encode("utf-8")).hexdigest()
    d = _patch_cache_dir()
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, f"{hid}_{kind}_{max_side}.png")


def _ensure_cropped_patch_resized(rgb_path: str, bbox: Tuple[int, int, int, int], max_side: int) -> str:
    outp = _cache_path_for_patch(rgb_path, bbox, "rgb_patch", max_side)
    if os.path.exists(outp):
        return outp
    img = open_rgb(rgb_path)
    w, h = img.size
    b = _clamp_bbox(bbox, w, h)
    crop = img.crop(b).convert("RGB")
    crop.thumbnail((max_side, max_side), Image.BICUBIC)
    crop.save(outp)
    return outp


def _ensure_cropped_nir_resized(nir_path: str, bbox: Tuple[int, int, int, int], max_side: int) -> str:
    outp = _cache_path_for_patch(nir_path, bbox, "nir_patch", max_side)
    if os.path.exists(outp):
        return outp
    nir = open_gray(nir_path)
    nir_rgb = nir_to_rgb(nir)
    w, h = nir_rgb.size
    b = _clamp_bbox(bbox, w, h)
    crop = nir_rgb.crop(b).convert("RGB")
    crop.thumbnail((max_side, max_side), Image.BICUBIC)
    crop.save(outp)
    return outp


# =========================
# Qwen3-VL multimodal messages
# =========================
def build_qwen3vl_messages(
    user_query: str,
    hits: List[RetrievalHit],
    query_image_path: Optional[str] = None,
    max_hits: int = 6,
    include_nir_evidence: bool = False,
    output_json: bool = False,
) -> List[Dict[str, Any]]:
    """
    关键点（为 6GB 显存保命）：
    - 不减少图像数量，但【统一缩放】待预测图像 + 证据图像，降低视觉 token。
    - 避免把“证据列表”在 prompt_text 里重复再写一遍（多模态下已经逐条给了 E# meta + 图）。
    - JSON 约束要严格但短，不要塞超长示例。
    """
    max_side = _vl_max_side()

    # 多模态时不再重复列证据元信息（避免 token 叠加）
    prompt_text = build_prompt(user_query, hits, max_hits=max_hits, include_evidence=False)
    if output_json:
        prompt_text += _json_schema_text()

    content: List[Dict[str, Any]] = []

    # 规则（短、强）
    content.append(
        {
            "type": "text",
            "text": (
                "执行规则：结论/理由必须引用证据编号 [E#]；若启用 JSON，必须只输出 JSON（无额外文本）。"
            ),
        }
    )

    # 待预测图像（缩放）
    if query_image_path:
        content.append({"type": "text", "text": "【待预测图像】"})
        if os.path.exists(query_image_path):
            q_img = _ensure_resized_rgb_image(query_image_path, max_side=max_side, kind="query")
            content.append({"type": "image", "image": _to_local_path(q_img)})
        else:
            # 兜底：仍然传原路径（不建议，但避免崩）
            content.append({"type": "image", "image": _to_local_path(query_image_path)})

    content.append({"type": "text", "text": "【检索证据图像】依次对应 [E1],[E2]...（每条先给元信息，再给图）"})

    # 证据图像（缩放 / patch 裁剪后再缩放）
    for i, h in enumerate(hits[:max_hits], 1):
        m = h.metadata or {}
        rgb_path = str(m.get("path", "") or "")
        nir_path = str(m.get("nir_path", "") or "") if include_nir_evidence else ""
        bbox_raw = m.get("bbox", None)
        typ = str(m.get("type", "") or "")

        # meta 行（尽量短，避免长路径）
        dist = getattr(h, "distance", None)
        try:
            dist_str = f"{float(dist):.4f}" if dist is not None else "NA"
        except Exception:
            dist_str = "NA"

        meta_parts = [
            f"[E{i}]",
            f"type={typ}",
            f"dist={dist_str}",
            f"tile_id={m.get('tile_id','')}",
            f"split={m.get('split','')}",
        ]
        if bbox_raw:
            meta_parts.append(f"bbox={bbox_raw}")
        labels = m.get("labels_present", None)
        if labels:
            meta_parts.append(f"labels_present={labels}")
        content.append({"type": "text", "text": " ".join([p for p in meta_parts if str(p).strip()])})

        bbox = _parse_bbox_any(bbox_raw) if (typ == "patch" and bbox_raw) else None

        # RGB evidence
        if rgb_path and os.path.exists(rgb_path):
            if bbox is not None:
                patch_path = _ensure_cropped_patch_resized(rgb_path, bbox, max_side=max_side)
                content.append({"type": "image", "image": _to_local_path(patch_path)})
            else:
                ev_path = _ensure_resized_rgb_image(rgb_path, max_side=max_side, kind=f"ev{i}")
                content.append({"type": "image", "image": _to_local_path(ev_path)})

        # NIR evidence（可选）
        if nir_path and os.path.exists(nir_path):
            content.append({"type": "text", "text": f"[E{i}] NIR（可选证据）"})
            if bbox is not None:
                nir_patch_path = _ensure_cropped_nir_resized(nir_path, bbox, max_side=max_side)
                content.append({"type": "image", "image": _to_local_path(nir_patch_path)})
            else:
                nir_ev = _ensure_resized_nir_rgb_image(nir_path, max_side=max_side, kind=f"nir{i}")
                content.append({"type": "image", "image": _to_local_path(nir_ev)})

    # 最后给任务与输出约束（短）
    content.append({"type": "text", "text": prompt_text})

    return [{"role": "user", "content": content}]


# =========================
# Rerank prompt (keep; slight compact)
# =========================
def _parse_labels_present(labels_present: str) -> List[str]:
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
    rerank 通常只走文本小模型/文本推理，显存压力小；这里保持强约束但尽量简短。
    """
    cand = hits[:max_hits]
    k = len(cand)
    eids = [f"E{i}" for i in range(1, k + 1)]

    cand_labels: List[List[str]] = []
    label_set = set()
    for h in cand:
        m = h.metadata or {}
        lbs = _parse_labels_present(str(m.get("labels_present", "") or ""))
        cand_labels.append(lbs)
        for lb in lbs:
            label_set.add(lb)

    labels_all = sorted(label_set)

    lines: List[str] = []
    lines.append("你是 rerank 助手。只输出一个 JSON：{ \"ranking\": [ ... ] }（无额外文本/无 Markdown）。")
    lines.append(f"必须输出恰好 {k} 条，覆盖全部 eid：{', '.join(eids)}；每个 eid 仅出现一次；score∈[0,1]。")
    lines.append("score 表示与用户问题的语义相关性（主要看 labels_present 等元信息）；sim_rank 仅用于并列时轻微区分。")
    lines.append("")
    lines.append(f"用户问题：{user_query}")
    if labels_all:
        lines.append("候选可能涉及的 labels（仅供理解，不要编造集合外标签）：")
        lines.append(", ".join(labels_all))
    lines.append("")
    lines.append("候选：")
    for i, h in enumerate(cand, 1):
        m = h.metadata or {}
        tile_id = str(m.get("tile_id", "") or "")
        split = str(m.get("split", "") or "")
        lbs = cand_labels[i - 1]
        lbs_text = "|".join(lbs) if lbs else ""
        lines.append(f"- E{i}: sim_rank={i}/{k}, tile_id={tile_id}, split={split}, labels_present={lbs_text}")

    lines.append("")
    lines.append("输出 schema：{ \"ranking\": [ {\"eid\":\"E1\",\"score\":0.0,\"reason\":\"...\"} ] }")

    content = [{"type": "text", "text": "\n".join(lines)}]
    return [{"role": "user", "content": content}]
