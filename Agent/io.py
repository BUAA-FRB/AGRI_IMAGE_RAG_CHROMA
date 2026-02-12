import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from .utils import read_json, try_float


@dataclass
class FinalOutput:
    schema: str
    id: str
    time_utc: str
    parse_ok: bool
    parsed_json: Dict[str, Any]
    normalized: Dict[str, Any]
    evidence: Dict[str, Any]
    evidence_order: List[str]
    query_image_path: Optional[str]
    assets: Dict[str, Any]
    src_json_path: str


def _pick(d: Dict[str, Any], *keys: str, default=None):
    for k in keys:
        if k in d:
            return d[k]
    return default


def load_final_output(json_path: str) -> FinalOutput:
    obj = read_json(json_path)

    schema = str(obj.get("schema", ""))
    fid = str(obj.get("id", os.path.splitext(os.path.basename(json_path))[0]))
    time_utc = str(obj.get("time_utc", ""))
    parse_ok = bool(obj.get("parse_ok", False))

    parsed_json = obj.get("parsed_json") or {}
    normalized = obj.get("normalized") or {}
    evidence = obj.get("evidence") or (obj.get("evidence_context", {}).get("evidence") or {})
    evidence_order = obj.get("evidence_order") or (obj.get("evidence_context", {}).get("evidence_order") or [])
    query_image_path = obj.get("query_image_path") or obj.get("evidence_context", {}).get("query_image_path")

    assets = obj.get("assets") or {}

    return FinalOutput(
        schema=schema,
        id=fid,
        time_utc=time_utc,
        parse_ok=parse_ok,
        parsed_json=parsed_json,
        normalized=normalized,
        evidence=evidence,
        evidence_order=list(evidence_order) if isinstance(evidence_order, list) else [],
        query_image_path=query_image_path,
        assets=assets,
        src_json_path=json_path,
    )


def get_primary_prediction(fo: FinalOutput) -> Tuple[str, float, List[str], str]:
    """
    returns: (label, confidence, refs, reason)
    """
    labels = fo.normalized.get("predicted_labels") or fo.parsed_json.get("predicted_labels") or []
    if isinstance(labels, list) and labels:
        # pick max confidence if present
        best = None
        best_c = -1.0
        for it in labels:
            if not isinstance(it, dict):
                continue
            c = try_float(it.get("confidence"), 0.0) or 0.0
            if c > best_c:
                best = it
                best_c = c
        if best:
            return (
                str(best.get("label", "unknown")),
                float(try_float(best.get("confidence"), 0.0) or 0.0),
                list(best.get("refs") or []),
                str(best.get("reason") or ""),
            )

    # fallback: normalized.labels
    norm_labels = fo.normalized.get("labels") or []
    if isinstance(norm_labels, list) and norm_labels:
        return str(norm_labels[0]), 0.0, [], ""
    return "unknown", 0.0, [], ""


def resolve_assets_root(fo: FinalOutput) -> str:
    """
    你的 JSON 里 assets.output_dir 是绝对路径（示例里是 E:\\...\\output）。
    若缺失，则用 json 文件所在目录作为根。
    """
    out_dir = fo.assets.get("output_dir")
    if isinstance(out_dir, str) and out_dir.strip():
        return out_dir
    return os.path.dirname(os.path.abspath(fo.src_json_path))


def resolve_asset_path(fo: FinalOutput, rel_or_abs: Optional[str]) -> Optional[str]:
    if not rel_or_abs:
        return None
    p = str(rel_or_abs)
    if os.path.isabs(p) and os.path.exists(p):
        return p
    root = resolve_assets_root(fo)
    cand = os.path.join(root, p)
    return cand
