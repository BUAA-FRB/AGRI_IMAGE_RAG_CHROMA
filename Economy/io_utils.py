from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


def read_json(path: str | Path) -> Dict[str, Any]:
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: str | Path, obj: Any) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def write_text(path: str | Path, text: str) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")


def relpath_for_md(target_path: str | Path, base_dir: str | Path) -> str:
    """
    Return a posix-like relative path for Markdown embedding.
    """
    tp = Path(target_path)
    bd = Path(base_dir)
    try:
        rp = os.path.relpath(tp, bd)
    except Exception:
        rp = str(tp)
    return rp.replace("\\", "/")


def parse_tile_id_from_path(path_str: str) -> Optional[str]:
    """
    Parse tile_id from paths like:
      datasets/.../field_images/rgb/<tile_id>.jpg
    """
    if not path_str:
        return None
    s = str(path_str).replace("\\", "/")
    name = s.split("/")[-1]
    if "." in name:
        stem = name.rsplit(".", 1)[0]
        if "_" in stem and "-" in stem:
            return stem
    return None


def pick_best_evidence_tile(latest: Dict[str, Any]) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[str]]:
    """
    Select tile_id from evidence_context by minimum distance.
    Returns: (tile_id, split, year, source_path)
    """
    ev_ctx = latest.get("evidence_context") or {}
    evidence = ev_ctx.get("evidence") or {}
    best = None
    best_dist = None
    for _, v in evidence.items():
        try:
            dist = float(v.get("distance"))
        except Exception:
            continue
        if best is None or (best_dist is not None and dist < best_dist):
            best = v
            best_dist = dist
    if not best:
        return (None, None, None, None)
    return (
        best.get("tile_id"),
        best.get("split"),
        str(best.get("year")) if best.get("year") is not None else None,
        best.get("source_path"),
    )


def find_stats_entry(field_stats: Dict[str, Any], tile_id: str) -> Tuple[Optional[str], Optional[str], Optional[Dict[str, Any]]]:
    """
    field_stats:
      { "train": { "<path_to_field_bounds_png>": {...} }, "val": {...}, "test": {...} }

    Match by suffix '/<tile_id>.png'
    Returns: (split_name, matched_key, entry_dict)
    """
    suffix = f"/{tile_id}.png"
    for split_name in ("train", "val", "test"):
        split_map = field_stats.get(split_name) or {}
        for k, v in split_map.items():
            ks = str(k).replace("\\", "/")
            if ks.endswith(suffix):
                return split_name, k, v
    return None, None, None


def safe_get_assets_preview_paths(latest: Dict[str, Any]) -> Dict[str, str]:
    """
    Extract useful preview paths from latest.json assets:
      - query preview
      - best evidence preview (rgb + nir if exists)
    Returns dict with keys like: query_preview, evidence_preview, evidence_nir_preview
    """
    out: Dict[str, str] = {}
    assets = latest.get("assets") or {}
    q = assets.get("query") or {}
    if q.get("preview_path"):
        out["query_preview"] = str(q["preview_path"])

    # best evidence eid: take E1 by evidence_order[0] if exists, else any
    ev_order = latest.get("evidence_order") or (latest.get("evidence_context") or {}).get("evidence_order") or []
    best_eid = ev_order[0] if ev_order else None

    ev_assets = (assets.get("evidence") or {})
    if best_eid and best_eid in ev_assets:
        ev = ev_assets[best_eid]
        if ev.get("preview_path"):
            out["evidence_preview"] = str(ev["preview_path"])
        if ev.get("nir_preview_path"):
            out["evidence_nir_preview"] = str(ev["nir_preview_path"])
    else:
        # fallback: pick any
        for _, ev in ev_assets.items():
            if ev.get("preview_path"):
                out["evidence_preview"] = str(ev["preview_path"])
                if ev.get("nir_preview_path"):
                    out["evidence_nir_preview"] = str(ev["nir_preview_path"])
                break

    return out
