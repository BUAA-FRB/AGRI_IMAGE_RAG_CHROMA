# agri_rag/structured_io.py
from __future__ import annotations

import json
import os
import re
import uuid
import shutil
import threading
import math
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Tuple, List

_CODE_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)

_TLS = threading.local()


# =========================
# Evidence context APIs (set by rag_predictor)
# =========================
def set_evidence_context_from_hits(
    hits: List[Any],
    query_image_path: Optional[str],
    max_hits: int = 6,
    include_nir_evidence: bool = False,
) -> None:
    """
    用 rag_predictor 的 hits 构造 E1..Ek 映射，供落盘 JSON / montage 使用。
    约定：E1 对应 hits[0]，E2 对应 hits[1] ... 与 prompt 中常见编号一致。
    """
    ev: Dict[str, Any] = {}
    order: List[str] = []
    use_hits = hits[: max_hits] if isinstance(hits, list) else []

    for i, h in enumerate(use_hits, start=1):
        eid = f"E{i}"
        order.append(eid)
        md = getattr(h, "metadata", None) or {}
        ev[eid] = {
            "eid": eid,
            "distance": getattr(h, "distance", None),
            "id": getattr(h, "id", None),
            "type": md.get("type", ""),
            "tile_id": md.get("tile_id", ""),
            "split": md.get("split", ""),
            "year": md.get("year", ""),
            "source_path": md.get("path", ""),
            "nir_path": md.get("nir_path", "") if include_nir_evidence else (md.get("nir_path", "") or ""),
            "bbox": md.get("bbox", ""),
            "labels_present": md.get("labels_present", ""),
        }

    ctx = {
        "schema": "agri_rag.evidence_context.v1",
        "time_utc": _utc_now_iso(),
        "query_image_path": os.path.abspath(query_image_path) if query_image_path else "",
        "include_nir_evidence": bool(include_nir_evidence),
        "evidence_order": order,
        "evidence": ev,
    }
    _TLS.evidence_ctx = ctx


def get_evidence_context() -> Optional[Dict[str, Any]]:
    ctx = getattr(_TLS, "evidence_ctx", None)
    if isinstance(ctx, dict):
        return ctx
    return None


def clear_evidence_context() -> None:
    if hasattr(_TLS, "evidence_ctx"):
        delattr(_TLS, "evidence_ctx")


def get_last_dump_paths() -> Dict[str, str]:
    """
    给 cli 用：打印最后一次落盘的 json / montage 路径。
    """
    d = getattr(_TLS, "last_dump_paths", None)
    return d if isinstance(d, dict) else {}


def _set_last_dump_paths(d: Dict[str, str]) -> None:
    _TLS.last_dump_paths = d


# =========================
# Basic text cleanup helpers
# =========================
def _strip_code_fence(text: str) -> str:
    m = _CODE_FENCE_RE.search(text)
    if m:
        return m.group(1).strip()
    return text.strip()


def _remove_trailing_commas(s: str) -> str:
    s = re.sub(r",\s*}", "}", s)
    s = re.sub(r",\s*]", "]", s)
    return s


def _strip_common_prefixes(s: str) -> str:
    t = s.strip()
    for pfx in ["JSON:", "json:", "Output:", "output:", "结果:", "输出:"]:
        if t.startswith(pfx):
            return t[len(pfx):].strip()
    return t


def _try_json_loads_dict_or_inner_dict(s: str) -> Optional[Dict[str, Any]]:
    try:
        obj = json.loads(s)
    except Exception:
        return None

    if isinstance(obj, dict):
        return obj

    if isinstance(obj, str):
        inner = obj.strip()
        if inner.startswith("{") and inner.endswith("}"):
            try:
                inner2 = _remove_trailing_commas(inner)
                obj2 = json.loads(inner2)
                if isinstance(obj2, dict):
                    return obj2
            except Exception:
                return None

    return None


def _scan_first_balanced_json_object(t: str) -> Optional[Dict[str, Any]]:
    start = -1
    depth = 0
    in_str = False
    esc = False

    for i, ch in enumerate(t):
        if in_str:
            if esc:
                esc = False
                continue
            if ch == "\\":
                esc = True
                continue
            if ch == '"':
                in_str = False
            continue

        if ch == '"':
            in_str = True
            continue

        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
            continue

        if ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start != -1:
                    cand = t[start : i + 1].strip()
                    cand = _remove_trailing_commas(cand)
                    obj = _try_json_loads_dict_or_inner_dict(cand)
                    if obj is not None:
                        return obj
                    start = -1
            continue

    return None


def _parse_json_object_core(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None

    t = text.replace("\ufeff", "").replace("\u200b", "")
    t = _strip_code_fence(t).strip()
    t = _strip_common_prefixes(t)
    t = t.strip()

    obj = _try_json_loads_dict_or_inner_dict(_remove_trailing_commas(t))
    if obj is not None:
        return obj

    l = t.find("{")
    r = t.rfind("}")
    if l != -1 and r != -1 and r > l:
        cand = t[l : r + 1].strip()
        cand = _remove_trailing_commas(cand)
        obj = _try_json_loads_dict_or_inner_dict(cand)
        if obj is not None:
            return obj

    obj = _scan_first_balanced_json_object(t)
    if obj is not None:
        return obj

    return None


# =========================
# Output dump helpers
# =========================
def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _is_rerank_like(obj: Dict[str, Any]) -> bool:
    if not isinstance(obj, dict):
        return False
    if "ranking" in obj and not any(k in obj for k in ("predicted_labels", "conclusion", "uncertainty")):
        return True
    return False


def _is_final_prediction_like(obj: Dict[str, Any]) -> bool:
    if not isinstance(obj, dict):
        return False
    if _is_rerank_like(obj):
        return False
    return any(k in obj for k in ("predicted_labels", "conclusion", "uncertainty"))


def _safe_float(x: Any) -> Optional[float]:
    if x is None:
        return None
    try:
        v = float(x)
        if v != v:
            return None
        return v
    except Exception:
        return None


def _clamp01(v: Optional[float]) -> Optional[float]:
    if v is None:
        return None
    if v < 0.0:
        return 0.0
    if v > 1.0:
        return 1.0
    return v


def _as_list_str(x: Any) -> list[str]:
    if x is None:
        return []
    if isinstance(x, list):
        out: list[str] = []
        for it in x:
            if isinstance(it, str) and it.strip():
                out.append(it.strip())
        return out
    if isinstance(x, str) and x.strip():
        return [x.strip()]
    return []


def dedup_keep_order(xs: list[str]) -> list[str]:
    seen = set()
    out: list[str] = []
    for x in xs:
        if x not in seen:
            out.append(x)
            seen.add(x)
    return out


def _normalize_predicted_labels_full(obj: Dict[str, Any]) -> Tuple[list[Dict[str, Any]], list[str]]:
    out_items: list[Dict[str, Any]] = []
    labels_linear: list[str] = []

    pls = obj.get("predicted_labels", None)

    if isinstance(pls, list):
        for it in pls:
            if isinstance(it, dict):
                lb = it.get("label", None)
                lb = lb.strip() if isinstance(lb, str) else ""
                if not lb:
                    continue
                conf = _clamp01(_safe_float(it.get("confidence", None)))
                refs = _as_list_str(it.get("refs", None))
                reason = it.get("reason", "")
                if not isinstance(reason, str):
                    reason = str(reason)

                out_items.append(
                    {"label": lb, "confidence": conf if conf is not None else None, "refs": refs, "reason": reason.strip()}
                )
                labels_linear.append(lb)

            elif isinstance(it, str) and it.strip():
                lb = it.strip()
                out_items.append({"label": lb, "confidence": None, "refs": [], "reason": ""})
                labels_linear.append(lb)

    if not labels_linear:
        conc = obj.get("conclusion", None)
        if isinstance(conc, str) and conc.strip():
            parts = re.split(r"[;,，、|/\n]+", conc)
            parts = [p.strip() for p in parts if p.strip()]
            labels_linear = dedup_keep_order(parts)
            for lb in labels_linear:
                out_items.append({"label": lb, "confidence": None, "refs": [], "reason": ""})

    labels_linear = dedup_keep_order([x for x in labels_linear if isinstance(x, str) and x.strip()])
    return out_items, labels_linear


def _output_dir() -> str:
    return os.path.abspath(os.getenv("AGRI_RAG_OUTPUT_DIR", "output"))


def _ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def _write_json(path: str, obj: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=str)

def _to_datasets_relpath(p: Any, anchor: str = "datasets") -> str:
    """
    把任意路径转换成从 datasets/ 开始的相对路径（posix 风格）。
    例：
      E:\\proj\\datasets\\data2018\\a.png -> datasets/data2018/a.png
      ./datasets/data2018/a.png          -> datasets/data2018/a.png
    若找不到 anchor，则退化为相对当前工作目录的相对路径（仍为 posix 风格）。
    """
    if p is None:
        return ""
    s = str(p).strip()
    if not s:
        return ""

    # 统一分隔符
    s2 = s.replace("\\", "/")

    # 已经是 datasets/... 或 ./datasets/...
    low = s2.lower()
    anchor_low = anchor.lower().strip("/")

    if low.startswith(f"{anchor_low}/"):
        return s2
    if low.startswith(f"./{anchor_low}/"):
        return s2[2:]  # 去掉 "./"

    # 在任意位置查找 "/datasets/" 或 "datasets/"
    needle1 = f"/{anchor_low}/"
    idx = low.find(needle1)
    if idx != -1:
        return s2[idx + 1 :]  # 去掉前导 "/"
    idx2 = low.find(f"{anchor_low}/")
    if idx2 != -1:
        return s2[idx2:]

    # 找不到 datasets：退化为相对 cwd 的路径
    try:
        rel = os.path.relpath(s, os.getcwd()).replace("\\", "/")
    except Exception:
        rel = s2

    if rel.startswith("./"):
        rel = rel[2:]
    return rel


def _rewrite_payload_dataset_paths_inplace(payload: Dict[str, Any], anchor: str = "datasets") -> Dict[str, Any]:
    """
    仅把最终输出 JSON 里与“数据集原图”相关的路径字段改成 datasets/... 相对路径：
      - payload["query_image_path"]
      - payload["evidence_context"]["query_image_path"]
      - payload["evidence_context"]["evidence"][eid]["source_path"/"nir_path"]
      - payload["evidence"][eid]["source_path"/"nir_path"]
      - payload["assets"]["query"]["source_path"]
      - payload["assets"]["evidence"][eid]["source_path"/"nir_path"]

    注意：preview_path / montage_path 本来就不是数据集原图路径，这里不改它们。
    """
    if not isinstance(payload, dict):
        return payload

    # root: query_image_path
    if "query_image_path" in payload:
        payload["query_image_path"] = _to_datasets_relpath(payload.get("query_image_path", ""), anchor=anchor)

    # evidence_context
    ctx = payload.get("evidence_context", None)
    if isinstance(ctx, dict):
        if "query_image_path" in ctx:
            ctx["query_image_path"] = _to_datasets_relpath(ctx.get("query_image_path", ""), anchor=anchor)

        ev = ctx.get("evidence", None)
        if isinstance(ev, dict):
            for _eid, it in ev.items():
                if not isinstance(it, dict):
                    continue
                if "source_path" in it:
                    it["source_path"] = _to_datasets_relpath(it.get("source_path", ""), anchor=anchor)
                if "nir_path" in it:
                    it["nir_path"] = _to_datasets_relpath(it.get("nir_path", ""), anchor=anchor)

    # payload["evidence"]（你在 _normalize_final_payload 里复制了一份）
    ev2 = payload.get("evidence", None)
    if isinstance(ev2, dict):
        for _eid, it in ev2.items():
            if not isinstance(it, dict):
                continue
            if "source_path" in it:
                it["source_path"] = _to_datasets_relpath(it.get("source_path", ""), anchor=anchor)
            if "nir_path" in it:
                it["nir_path"] = _to_datasets_relpath(it.get("nir_path", ""), anchor=anchor)

    # assets
    assets = payload.get("assets", None)
    if isinstance(assets, dict):
        q = assets.get("query", None)
        if isinstance(q, dict) and "source_path" in q:
            q["source_path"] = _to_datasets_relpath(q.get("source_path", ""), anchor=anchor)

        aev = assets.get("evidence", None)
        if isinstance(aev, dict):
            for _eid, it in aev.items():
                if not isinstance(it, dict):
                    continue
                if "source_path" in it:
                    it["source_path"] = _to_datasets_relpath(it.get("source_path", ""), anchor=anchor)
                if "nir_path" in it:
                    it["nir_path"] = _to_datasets_relpath(it.get("nir_path", ""), anchor=anchor)

    return payload


def _relpath_under(out_dir: str, p: str) -> str:
    try:
        ap = os.path.abspath(p)
        ao = os.path.abspath(out_dir)
        if ap.startswith(ao):
            return os.path.relpath(ap, ao).replace("\\", "/")
    except Exception:
        pass
    return p.replace("\\", "/")


def _parse_bbox(bbox_str: Any) -> Optional[Tuple[int, int, int, int]]:
    if not bbox_str:
        return None
    if isinstance(bbox_str, (tuple, list)) and len(bbox_str) == 4:
        try:
            a = [int(float(x)) for x in bbox_str]
            return a[0], a[1], a[2], a[3]
        except Exception:
            return None
    if not isinstance(bbox_str, str):
        return None
    nums = re.findall(r"-?\d+(?:\.\d+)?", bbox_str)
    if len(nums) >= 4:
        try:
            a = [int(float(nums[i])) for i in range(4)]
            return a[0], a[1], a[2], a[3]
        except Exception:
            return None
    return None


def _load_pil_image(path: str):
    try:
        from PIL import Image
        return Image.open(path)
    except Exception:
        return None


def _save_preview_image(src_img, out_path: str, max_side: int = 512) -> Optional[Tuple[int, int]]:
    """
    保存预览图（缩放到 max_side），返回 (w,h)。
    """
    try:
        from PIL import Image
        img = src_img.copy()
        img = img.convert("RGB")
        w, h = img.size
        scale = min(1.0, float(max_side) / float(max(w, h)))
        if scale < 1.0:
            nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
            img = img.resize((nw, nh), Image.BILINEAR)
        _ensure_dir(os.path.dirname(out_path))
        img.save(out_path, format="PNG")
        return img.size
    except Exception:
        return None


def _materialize_assets_and_montage(payload: Dict[str, Any], out_dir: str, base_name: str) -> Dict[str, Any]:
    """
    - 输出 query + evidence 的 preview 图到 output/assets/
    - 输出 montage 到 output/
    - 并把路径写入 payload["assets"] / payload["montage"]
    """
    assets_dir = os.path.join(out_dir, "assets")
    _ensure_dir(assets_dir)

    ctx = payload.get("evidence_context", None)
    if not isinstance(ctx, dict):
        return payload

    query_path = ctx.get("query_image_path", "") or ""
    evidence = ctx.get("evidence", {}) or {}
    order = ctx.get("evidence_order", []) or []

    assets: Dict[str, Any] = {
        "output_dir": out_dir,
        "assets_dir": _relpath_under(out_dir, assets_dir),
        "query": None,
        "evidence": {},
        "montage": None,
    }

    thumbs: List[Tuple[str, Any]] = []  # (label, PIL.Image)
    thumb_size = int(os.getenv("AGRI_RAG_MONTAGE_THUMB", "384"))
    gap = int(os.getenv("AGRI_RAG_MONTAGE_GAP", "12"))

    # Query preview
    if query_path and os.path.exists(query_path):
        img = _load_pil_image(query_path)
        if img is not None:
            q_out = os.path.join(assets_dir, f"{base_name}_query.png")
            size = _save_preview_image(img, q_out, max_side=512)
            assets["query"] = {
                "source_path": os.path.abspath(query_path),
                "preview_path": _relpath_under(out_dir, q_out),
                "size": {"w": size[0], "h": size[1]} if size else None,
            }
            # montage thumb
            try:
                from PIL import Image
                timg = img.convert("RGB")
                timg.thumbnail((thumb_size, thumb_size), Image.BILINEAR)
                thumbs.append(("QUERY", timg))
            except Exception:
                pass

    # Evidence previews
    for eid in order:
        it = evidence.get(eid, None)
        if not isinstance(it, dict):
            continue

        src_path = it.get("source_path", "") or ""
        nir_path = it.get("nir_path", "") or ""
        bbox = _parse_bbox(it.get("bbox", ""))

        ev_item: Dict[str, Any] = {
            "eid": eid,
            "type": it.get("type", ""),
            "distance": it.get("distance", None),
            "tile_id": it.get("tile_id", ""),
            "split": it.get("split", ""),
            "year": it.get("year", ""),
            "bbox": it.get("bbox", ""),
            "source_path": os.path.abspath(src_path) if src_path else "",
            "nir_path": os.path.abspath(nir_path) if nir_path else "",
            "preview_path": "",
            "nir_preview_path": "",
            "preview_size": None,
            "nir_preview_size": None,
            "preview_mode": "full",
        }

        # RGB preview (crop if patch bbox available)
        if src_path and os.path.exists(src_path):
            img = _load_pil_image(src_path)
            if img is not None:
                use_img = img
                if bbox is not None and (it.get("type", "") == "patch" or it.get("bbox", "")):
                    try:
                        use_img = img.crop(bbox)
                        ev_item["preview_mode"] = "crop"
                        ev_item["bbox_parsed"] = {"x0": bbox[0], "y0": bbox[1], "x1": bbox[2], "y1": bbox[3]}
                    except Exception:
                        use_img = img
                        ev_item["preview_mode"] = "full"

                outp = os.path.join(assets_dir, f"{base_name}_{eid}.png")
                size = _save_preview_image(use_img, outp, max_side=512)
                ev_item["preview_path"] = _relpath_under(out_dir, outp)
                if size:
                    ev_item["preview_size"] = {"w": size[0], "h": size[1]}

                # montage thumb from rgb preview
                try:
                    from PIL import Image
                    timg = use_img.convert("RGB")
                    timg.thumbnail((thumb_size, thumb_size), Image.BILINEAR)
                    thumbs.append((eid, timg))
                except Exception:
                    pass

        # NIR preview (optional)
        if nir_path and os.path.exists(nir_path):
            nimg = _load_pil_image(nir_path)
            if nimg is not None:
                use_n = nimg
                if bbox is not None and (it.get("type", "") == "patch" or it.get("bbox", "")):
                    try:
                        use_n = nimg.crop(bbox)
                    except Exception:
                        use_n = nimg
                # grayscale -> rgb
                try:
                    use_n = use_n.convert("L").convert("RGB")
                except Exception:
                    pass

                outn = os.path.join(assets_dir, f"{base_name}_{eid}_nir.png")
                nsize = _save_preview_image(use_n, outn, max_side=512)
                ev_item["nir_preview_path"] = _relpath_under(out_dir, outn)
                if nsize:
                    ev_item["nir_preview_size"] = {"w": nsize[0], "h": nsize[1]}

        assets["evidence"][eid] = ev_item

    # Montage output (query + evidence previews)
    if os.getenv("AGRI_RAG_DISABLE_MONTAGE", "").strip() not in ("1", "true", "TRUE", "yes", "YES") and thumbs:
        try:
            from PIL import Image, ImageDraw

            n = len(thumbs)
            cols = int(os.getenv("AGRI_RAG_MONTAGE_COLS", "0"))
            if cols <= 0:
                cols = max(2, min(4, int(math.ceil(math.sqrt(n)))))
            rows = int(math.ceil(n / float(cols)))

            cell = thumb_size
            W = cols * cell + (cols + 1) * gap
            H = rows * cell + (rows + 1) * gap + 30  # header
            canvas = Image.new("RGB", (W, H), (255, 255, 255))
            draw = ImageDraw.Draw(canvas)

            # header
            draw.text((gap, 8), "QUERY + EVIDENCE (E1..)", fill=(0, 0, 0))

            for idx, (label, img) in enumerate(thumbs):
                r = idx // cols
                c = idx % cols
                x0 = gap + c * (cell + gap)
                y0 = 30 + gap + r * (cell + gap)

                # center pad
                tile = Image.new("RGB", (cell, cell), (245, 245, 245))
                iw, ih = img.size
                ox = (cell - iw) // 2
                oy = (cell - ih) // 2
                tile.paste(img, (ox, oy))

                canvas.paste(tile, (x0, y0))
                draw.text((x0 + 4, y0 + 4), str(label), fill=(0, 0, 0))

            montage_path = os.path.join(out_dir, f"montage_{base_name}.png")
            canvas.save(montage_path, format="PNG")

            # latest montage
            latest_montage = os.path.join(out_dir, "latest_montage.png")
            try:
                shutil.copyfile(montage_path, latest_montage)
            except Exception:
                pass

            assets["montage"] = {
                "path": _relpath_under(out_dir, montage_path),
                "latest_path": _relpath_under(out_dir, latest_montage),
                "size": {"w": canvas.size[0], "h": canvas.size[1]},
            }
        except Exception:
            pass

    payload["assets"] = assets
    return payload


def _normalize_final_payload(raw_text: str, obj: Dict[str, Any], parse_ok: bool = True) -> Dict[str, Any]:
    pred_items, labels = _normalize_predicted_labels_full(obj)

    conclusion = obj.get("conclusion", "")
    if not isinstance(conclusion, str):
        conclusion = str(conclusion)

    uncertainty = obj.get("uncertainty", "")
    if uncertainty is None:
        uncertainty = ""
    if not isinstance(uncertainty, str):
        uncertainty = str(uncertainty)

    payload: Dict[str, Any] = {
        "schema": "agri_rag.final_output.v2",
        "time_utc": _utc_now_iso(),
        "id": uuid.uuid4().hex,
        "parse_ok": bool(parse_ok),
        "raw_text": raw_text,
        "parsed_json": obj,
        "normalized": {
            "predicted_labels": pred_items,
            "labels": labels,
            "conclusion": conclusion.strip(),
            "uncertainty": uncertainty.strip(),
        },
    }

    # attach evidence context if present
    ctx = get_evidence_context()
    if ctx is not None:
        payload["evidence_context"] = ctx
        payload["evidence_order"] = ctx.get("evidence_order", [])
        payload["evidence"] = ctx.get("evidence", {})
        payload["query_image_path"] = ctx.get("query_image_path", "")

    return payload


def _auto_dump_final_if_needed(raw_text: str, obj: Dict[str, Any], parse_ok: bool = True) -> Optional[str]:
    if os.getenv("AGRI_RAG_DISABLE_AUTO_DUMP", "").strip() in ("1", "true", "TRUE", "yes", "YES"):
        return None

    if not _is_final_prediction_like(obj):
        return None

    out_dir = _output_dir()
    _ensure_dir(out_dir)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    sid = uuid.uuid4().hex[:8]
    base_name = f"{ts}_{sid}"

    json_name = f"final_{base_name}.json"
    json_path = os.path.join(out_dir, json_name)

    payload = _normalize_final_payload(raw_text=raw_text, obj=obj, parse_ok=parse_ok)

    # NEW: materialize evidence preview images + montage, and save paths into payload
    try:
        payload = _materialize_assets_and_montage(payload, out_dir=out_dir, base_name=base_name)
    except Exception:
        pass

    # NEW: rewrite dataset image paths (abs -> datasets/...)
    try:
        payload = _rewrite_payload_dataset_paths_inplace(payload, anchor="datasets")
    except Exception:
        pass

    _write_json(json_path, payload)


    latest_path = os.path.join(out_dir, "latest.json")
    try:
        _write_json(latest_path, payload)
    except Exception:
        pass

    # record paths for cli
    d: Dict[str, str] = {
        "json_path": os.path.abspath(json_path),
        "latest_json": os.path.abspath(latest_path),
    }
    assets = payload.get("assets", {}) if isinstance(payload, dict) else {}
    if isinstance(assets, dict):
        mon = assets.get("montage", None)
        if isinstance(mon, dict):
            p1 = mon.get("path", "")
            p2 = mon.get("latest_path", "")
            if p1:
                d["montage_path"] = os.path.abspath(os.path.join(out_dir, p1.replace("/", os.sep)))
            if p2:
                d["latest_montage"] = os.path.abspath(os.path.join(out_dir, p2.replace("/", os.sep)))

    _set_last_dump_paths(d)
    return json_path


# =========================
# Public APIs
# =========================
def parse_json_object(text: str) -> Optional[Dict[str, Any]]:
    """
    仅解析 JSON，不做落盘（给 verify 用，避免重复输出文件）。
    """
    return _parse_json_object_core(text)


def extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    """
    解析模型输出 JSON，并自动落盘“最终预测结果”到 output/：
      - output/final_<timestamp>_<sid>.json
      - output/latest.json
      - output/assets/...(query/evidence preview)
      - output/montage_<...>.png & output/latest_montage.png

    注意：
      - 如果模型输出不是 JSON，但包含自然语言，我们会 fallback 成 {"conclusion": raw_text}
        仍然落盘（方便你后续展示 raw_text + evidence）。
    """
    if not text:
        return None

    raw_text = text

    obj = _parse_json_object_core(text)
    if obj is not None:
        _auto_dump_final_if_needed(raw_text=raw_text, obj=obj, parse_ok=True)
        return obj

    # fallback: dump as conclusion-only
    fallback = {"conclusion": raw_text.strip(), "predicted_labels": [], "uncertainty": ""}
    _auto_dump_final_if_needed(raw_text=raw_text, obj=fallback, parse_ok=False)
    return None


def normalize_predicted_labels(obj: Dict[str, Any]) -> list[str]:
    out: list[str] = []
    if not isinstance(obj, dict):
        return out

    pls = obj.get("predicted_labels", None)
    if isinstance(pls, list):
        for it in pls:
            if isinstance(it, dict):
                lb = it.get("label", None)
                if isinstance(lb, str) and lb.strip():
                    out.append(lb.strip())
            elif isinstance(it, str) and it.strip():
                out.append(it.strip())

    if out:
        return dedup_keep_order(out)

    conc = obj.get("conclusion", None)
    if isinstance(conc, str) and conc.strip():
        parts = re.split(r"[;,，、|/\n]+", conc)
        parts = [p.strip() for p in parts if p.strip()]
        return dedup_keep_order(parts)

    return out
