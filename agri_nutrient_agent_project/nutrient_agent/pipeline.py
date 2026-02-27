from __future__ import annotations

import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

from .config import AgentConfig, LLMConfig
from .io_utils import copy_file, ensure_dir, pick_existing_path, read_json, safe_relpath, write_json
from .schema import (
    AreaStats,
    DecisionOut,
    DiagnosisOut,
    FrontendScene,
    FrontendTerrain,
    LayersOut,
    NutrientDeficiencyOutput,
    SuspectedNutrient,
)
from .tools.indices import vigor_proxy
from .tools.severity import compute_severity_from_vigor, connected_components
from .tools.geojson_utils import BBoxLngLat, bbox_polygon_geojson, rectangles_to_geojson, sampling_points_geojson
from .tools.render import (
    save_index_preview,
    save_severity_heatmap_rgba,
    save_low_vigor_histogram,
    save_severity_distribution_bar,
    save_patch_area_bar,
)
from .tools.report import generate_report_md
from .tools.image_utils import make_uniform_grid_montage, rgba_to_rgb_white

# Optional LLM
from .tools.llm_qwen import generate_bilingual_summary_with_qwen25, LLMResult


def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _rand_run_id(seed: str) -> str:
    h = hashlib.sha1(seed.encode("utf-8")).hexdigest()
    return h[:12]


def _is_nutrient_in_upstream(up: Dict[str, Any]) -> Tuple[bool, float]:
    norm = up.get("normalized") or {}
    labels = norm.get("labels") or []
    predicted = norm.get("predicted_labels") or (up.get("parsed_json") or {}).get("predicted_labels") or []
    conf = 0.0
    for item in predicted:
        if item.get("label") == "nutrient_deficiency":
            conf = max(conf, float(item.get("confidence", 0.0)))
    return ("nutrient_deficiency" in labels), conf


def _pick_query_image(up: Dict[str, Any], cli_query: Optional[Path]) -> Optional[Path]:
    if cli_query and cli_query.exists():
        return cli_query
    q = up.get("query_image_path") or (up.get("evidence_context") or {}).get("query_image_path") or ""
    if q:
        cands = [Path(q)]
        if "demo/" in q:
            cands.append(Path(q.replace("demo/", "Demo/")))
        if "Demo/" in q:
            cands.append(Path(q.replace("Demo/", "demo/")))
        p = pick_existing_path(cands)
        if p:
            return p
    return pick_existing_path([Path("Demo/query.jpg"), Path("demo/query.jpg")])


def _default_bbox(cfg: AgentConfig) -> BBoxLngLat:
    lng, lat = cfg.default_center_lnglat
    ext = cfg.default_extent_deg
    return BBoxLngLat(min_lng=lng - ext, min_lat=lat - ext, max_lng=lng + ext, max_lat=lat + ext)


def _extract_field_meta_bbox(field_meta: Dict[str, Any], cfg: AgentConfig) -> BBoxLngLat:
    if not field_meta:
        return _default_bbox(cfg)
    if isinstance(field_meta, dict) and "bbox" in field_meta:
        b = field_meta["bbox"]
        if isinstance(b, (list, tuple)) and len(b) == 4:
            return BBoxLngLat(min_lng=float(b[0]), min_lat=float(b[1]), max_lng=float(b[2]), max_lat=float(b[3]))
    try:
        coords = []
        if field_meta.get("type") == "FeatureCollection":
            feats = field_meta.get("features") or []
            for f in feats:
                g = f.get("geometry") or {}
                if g.get("type") == "Polygon":
                    coords.extend(g.get("coordinates")[0])
        elif field_meta.get("type") == "Feature":
            g = field_meta.get("geometry") or {}
            if g.get("type") == "Polygon":
                coords.extend(g.get("coordinates")[0])
        xs = [c[0] for c in coords]
        ys = [c[1] for c in coords]
        if xs and ys:
            return BBoxLngLat(min_lng=min(xs), min_lat=min(ys), max_lng=max(xs), max_lat=max(ys))
    except Exception:
        pass
    return _default_bbox(cfg)


def _build_sampling_points_from_components(
    comps: List[Tuple[int, int, int, int, int]],
    W: int,
    H: int,
    max_points: int,
) -> List[Tuple[int, int, str]]:
    pts: List[Tuple[int, int, str]] = []
    for i, (x0, y0, x1, y1, area) in enumerate(comps[:max_points]):
        cx = (x0 + x1) // 2
        cy = (y0 + y1) // 2
        pts.append((int(cx), int(cy), f"建议在斑块中心采样（patch#{i}, area_px={area}）"))
    if len(pts) < max_points:
        extras = [(W // 6, H // 6), (5 * W // 6, H // 6), (W // 6, 5 * H // 6), (5 * W // 6, 5 * H // 6)]
        for (x, y) in extras:
            if len(pts) >= max_points:
                break
            pts.append((int(x), int(y), "对照采样点（健康区/边缘区）"))
    return pts[:max_points]


def _collect_evidence_blocks(up: Dict[str, Any], out_assets_dir: Path, upstream_dir: Path) -> List[Dict[str, Any]]:
    blocks: List[Dict[str, Any]] = []
    assets = up.get("assets") or {}
    ev_assets = assets.get("evidence") or {}
    evidence = up.get("evidence") or {}
    order = up.get("evidence_order") or list(evidence.keys())

    for eid in order:
        ev = evidence.get(eid) or {}
        ev_a = ev_assets.get(eid) or {}
        blk: Dict[str, Any] = {"eid": eid, "labels_present": ev.get("labels_present", "")}

        prev = ev_a.get("preview_path") or ""
        nir_prev = ev_a.get("nir_preview_path") or ""

        if prev:
            src = upstream_dir / prev
            dst = out_assets_dir / "evidence" / f"{eid}.png"
            if copy_file(src, dst):
                blk["preview"] = safe_relpath(str(Path("assets") / "evidence" / f"{eid}.png"))

        if nir_prev:
            src = upstream_dir / nir_prev
            dst = out_assets_dir / "evidence" / f"{eid}_nir.png"
            if copy_file(src, dst):
                blk["nir_preview"] = safe_relpath(str(Path("assets") / "evidence" / f"{eid}_nir.png"))

        if "nutrient_deficiency" in (ev.get("labels_present") or ""):
            blk["note"] = "该证据样本在数据集中标注包含 nutrient_deficiency，可作为对照参考。"

        blocks.append(blk)

    return blocks


def _make_evidence_montage(assets_dir: Path, evidence_blocks: List[Dict[str, Any]]) -> Optional[str]:
    imgs: List[Image.Image] = []
    for blk in evidence_blocks[:6]:
        rel = blk.get("preview")
        if not rel:
            continue
        p = assets_dir.parent / rel
        if p.exists():
            try:
                imgs.append(Image.open(p))
            except Exception:
                pass
    if not imgs:
        return None
    montage = make_uniform_grid_montage(imgs, cols=3, rows=2, cell_size=(320, 320), pad=14, force_rgb=True)
    out_path = assets_dir / "evidence_montage.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    montage.save(out_path)
    return safe_relpath(str(Path("assets") / "evidence_montage.png"))


def _make_triptych_visuals(
    assets_dir: Path,
    query_preview_path: Path,
    index_png_path: Path,
    heatmap_png_path: Path,
) -> Optional[str]:
    """
    Create a 1×3 montage:
      [Query Preview] [Vigor Index] [Severity Heatmap]
    Ensure equal size + aligned.
    """
    imgs: List[Image.Image] = []
    if query_preview_path.exists():
        imgs.append(Image.open(query_preview_path))
    if index_png_path.exists():
        imgs.append(Image.open(index_png_path))
    if heatmap_png_path.exists():
        # heatmap may be RGBA; composite on white for montage
        imgs.append(rgba_to_rgb_white(Image.open(heatmap_png_path)))

    if len(imgs) != 3:
        return None

    trip = make_uniform_grid_montage(imgs, cols=3, rows=1, cell_size=(360, 360), pad=14, force_rgb=True)
    out_path = assets_dir / "visuals_triptych.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    trip.save(out_path)
    return safe_relpath(str(Path("assets") / "visuals_triptych.png"))


def run_nutrient_deficiency_agent(
    upstream_path: Path,
    query_image_path: Optional[Path],
    out_dir: Path,
    force: bool,
    llm_mode: str,
    qwen25_path: Path,
    qwen3vl_path: Path,
    field_meta_path: Optional[Path] = None,
) -> None:
    cfg = AgentConfig()
    llm_cfg = LLMConfig(mode=llm_mode, qwen25_path=qwen25_path, qwen3vl_path=qwen3vl_path)

    up = read_json(upstream_path)
    upstream_dir = upstream_path.parent

    is_nutr, up_conf = _is_nutrient_in_upstream(up)
    applicable = bool(is_nutr) or bool(force)

    qpath = _pick_query_image(up, query_image_path)
    if not qpath or not qpath.exists():
        raise FileNotFoundError(f"Query image not found. Provided={query_image_path}, upstream={up.get('query_image_path')}")

    ensure_dir(out_dir)
    assets_dir = out_dir / "assets"
    geojson_dir = out_dir / "geojson"
    ensure_dir(assets_dir)
    ensure_dir(geojson_dir)

    run_id = _rand_run_id(str(up.get("id", "")) + str(time.time()))

    # Copy query preview
    q_preview_rel: Optional[str] = None
    up_assets = up.get("assets") or {}
    q_prev = ((up_assets.get("query") or {}).get("preview_path") or "")
    query_preview_abs = assets_dir / "query_preview.png"

    if q_prev:
        src = upstream_dir / q_prev
        if copy_file(src, query_preview_abs):
            q_preview_rel = safe_relpath(str(Path("assets") / "query_preview.png"))
    if not q_preview_rel:
        copy_file(qpath, query_preview_abs)
        q_preview_rel = safe_relpath(str(Path("assets") / "query_preview.png"))

    # Load query and compute severity
    q_img = Image.open(qpath).convert("RGB")
    rgb01 = np.asarray(q_img).astype(np.float32) / 255.0
    H, W = rgb01.shape[:2]

    vigor = vigor_proxy(rgb01)
    sev = compute_severity_from_vigor(vigor, cfg.mild_pctl, cfg.moderate_pctl, cfg.severe_pctl)

    low_std = float(np.std(sev.low_vigor01))
    aff_ratio = float(np.mean(sev.affected_mask))
    agent_score = float(np.clip(0.35 + 2.2 * low_std + 0.8 * aff_ratio, 0.0, 1.0))

    comps = connected_components(sev.affected_mask, min_area=180)
    patch_count = len(comps)
    affected_area_px = int(np.sum(sev.affected_mask))
    largest_patch_px = int(comps[0][4]) if comps else 0

    # BBox
    field_meta: Dict[str, Any] = {}
    if field_meta_path and field_meta_path.exists():
        try:
            field_meta = read_json(field_meta_path)
        except Exception:
            field_meta = {}
    bbox_ll = _extract_field_meta_bbox(field_meta, cfg)

    # GeoJSON outputs
    sev_polys = rectangles_to_geojson(comps, W, H, bbox_ll, props_extra={"severity": "moderate_or_severe"})
    write_json(geojson_dir / "severity_polygons.geojson", sev_polys)

    pts_px = _build_sampling_points_from_components(comps, W, H, cfg.max_sampling_points)
    pts_gj = sampling_points_geojson(pts_px, W, H, bbox_ll)
    write_json(geojson_dir / "sampling_points.geojson", pts_gj)

    write_json(geojson_dir / "field_bbox.geojson", bbox_polygon_geojson(bbox_ll))

    # Render base visuals
    index_png_abs = assets_dir / "vigor_index.png"
    heat_png_abs = assets_dir / "severity_heatmap.png"
    save_index_preview(sev.vigor01, index_png_abs, title="Vigor proxy (RGB)")
    save_severity_heatmap_rgba(sev.class_map, heat_png_abs, alpha=0.7)

    # NEW: Triptych montage (Query/Index/Heatmap)
    triptych_rel = _make_triptych_visuals(
        assets_dir=assets_dir,
        query_preview_path=query_preview_abs,
        index_png_path=index_png_abs,
        heatmap_png_path=heat_png_abs,
    )

    # Evidence blocks & montage
    evidence_blocks = _collect_evidence_blocks(up, assets_dir, upstream_dir)
    evidence_montage_rel = _make_evidence_montage(assets_dir, evidence_blocks)

    # NEW: Charts
    hist_abs = assets_dir / "low_vigor_hist.png"
    dist_abs = assets_dir / "severity_dist.png"
    patch_bar_abs = assets_dir / "patch_area_bar.png"

    save_low_vigor_histogram(sev.low_vigor01, sev.thresholds, hist_abs, title="Low-vigor Histogram with Thresholds")
    severity_counts = save_severity_distribution_bar(sev.class_map, dist_abs, title="Severity Distribution (pixel counts)")
    _ = save_patch_area_bar(comps, patch_bar_abs, top_k=10, title="Top Patch Areas (px)")

    # top patch table entries
    top_patches = []
    for i, (x0, y0, x1, y1, area) in enumerate(comps[:10]):
        top_patches.append({"rank": i, "area_px": int(area), "bbox_px": f"({x0},{y0},{x1},{y1})"})

    # Suspected nutrients (heuristic)
    suspected = [
        SuspectedNutrient(nutrient="N", prob=0.50, evidence=["IDX:vigor_low"]),
        SuspectedNutrient(nutrient="Fe", prob=0.18, evidence=["IDX:vigor_low"]),
        SuspectedNutrient(nutrient="K", prob=0.12, evidence=["IDX:vigor_low"]),
    ]

    # Default bilingual summaries
    base_cn = "在 query 图像中检测到低活力（低绿度）斑块，可能与营养缺乏或水分胁迫等因素有关。建议优先通过采样与农事记录/天气信息进行复核，并制定分区管理措施。"
    base_en = "Low-vigor patches are detected in the query image. They may be related to nutrient deficiency or other stressors (e.g., water stress). Verify with sampling plus ops/weather context, then plan zone-based actions."

    uncertainty_notes: List[str] = []
    if not applicable:
        uncertainty_notes.append("上游未将该 query 判为 nutrient_deficiency；本 Agent 输出主要用于对照/回测或在 --force 模式下演示。")
    uncertainty_notes.append("仅基于 RGB 的活力 proxy 可能与干旱、病害等胁迫混淆；如有 NIR/多时相影像，可显著增强判别。")

    # Optional LLM narrative
    if llm_cfg.mode in ("qwen2.5", "auto") and qwen25_path.exists():
        try:
            prompt_cn = (
                "请根据以下结构化信息生成双语摘要，格式严格为：\n"
                "【中文摘要】...\n【English Summary】...\n【Uncertainty】逐条列出...\n\n"
                f"上游结论：{(up.get('normalized') or {}).get('conclusion','')}\n"
                f"上游标签：{(up.get('normalized') or {}).get('labels', [])}, nutrient_deficiency_conf={up_conf}\n"
                f"本 Agent 统计：affected_ratio={aff_ratio:.3f}, patch_count={patch_count}, low_vigor_std={low_std:.3f}\n"
                f"阈值：{sev.thresholds}\n"
                "要求：1) 不要给出具体化学品配方/用量；2) 建议以采样验证与农艺规范为主；3) 保留不确定性。"
            )
            llm_res: LLMResult = generate_bilingual_summary_with_qwen25(
                model_path=qwen25_path,
                prompt_cn=prompt_cn,
                max_new_tokens=llm_cfg.max_new_tokens,
                temperature=llm_cfg.temperature,
                top_p=llm_cfg.top_p,
                repetition_penalty=llm_cfg.repetition_penalty,
            )
            summary_cn = llm_res.summary_cn
            summary_en = llm_res.summary_en
            uncertainty_notes.extend(llm_res.uncertainty_notes)
        except Exception as e:
            summary_cn, summary_en = base_cn, base_en
            uncertainty_notes.append(f"LLM 生成失败，已回退到模板摘要：{type(e).__name__}")
    else:
        summary_cn, summary_en = base_cn, base_en

    actions_cn = [
        "在重度斑块与健康区各选若干点做叶片/土壤采样，验证N/P/K与pH/EC等关键指标。",
        "核对近期农事记录（施肥/灌溉/降雨），排查“水分胁迫/淋洗”导致的类似表征。",
        "按斑块分区管理：优先对重度区做补救措施，对轻度区加强监测与复拍。",
    ]
    actions_en = [
        "Sample leaves/soil in both severe patches and healthy zones to verify key nutrients and soil conditions (e.g., pH/EC).",
        "Cross-check recent operations (fertilization/irrigation) and rainfall to rule out water-stress or leaching effects.",
        "Use zone-based management: prioritize remediation in severe zones; monitor mild zones with follow-up imaging.",
    ]
    monitoring_cn = [
        "T+7 天：复拍同区域影像，对比活力指数/斑块面积变化。",
        "T+14 天：结合采样结果与复拍趋势，评估是否需要调整管理策略。",
    ]
    monitoring_en = [
        "T+7d: Re-image the same area and compare vigor indices and patch extent changes.",
        "T+14d: Combine lab results and trend signals to decide whether to adjust the management plan.",
    ]

    images = {
        "query": q_preview_rel,
        "index": safe_relpath(str(Path("assets") / "vigor_index.png")),
        "heatmap": safe_relpath(str(Path("assets") / "severity_heatmap.png")),
        "visuals_triptych": triptych_rel or "",
        "evidence_montage": evidence_montage_rel or "",
        "low_vigor_hist": safe_relpath(str(Path("assets") / "low_vigor_hist.png")),
        "severity_dist": safe_relpath(str(Path("assets") / "severity_dist.png")),
        "patch_area_bar": safe_relpath(str(Path("assets") / "patch_area_bar.png")),
    }

    report_md = generate_report_md(
        title_cn=cfg.report_title_cn,
        title_en=cfg.report_title_en,
        summary_cn=summary_cn,
        summary_en=summary_en,
        decision={
            "is_applicable": applicable,
            "upstream_label_conf": float(up_conf),
            "agent_validation_score": float(agent_score),
            "uncertainty_notes": uncertainty_notes,
        },
        area_stats={
            "affected_area_px": affected_area_px,
            "affected_ratio": aff_ratio,
            "patch_count": patch_count,
            "largest_patch_px": largest_patch_px,
        },
        thresholds=sev.thresholds,
        images=images,
        evidence_blocks=evidence_blocks,
        actions_cn=actions_cn,
        actions_en=actions_en,
        monitoring_cn=monitoring_cn,
        monitoring_en=monitoring_en,
        extra={
            "severity_counts": severity_counts,
            "top_patches": top_patches,
        },
    )
    (out_dir / "report.md").write_text(report_md, encoding="utf-8")

    # Scene center uses bbox center (avoid Beijing city placeholder)
    center_lng = (bbox_ll.min_lng + bbox_ll.max_lng) / 2.0
    center_lat = (bbox_ll.min_lat + bbox_ll.max_lat) / 2.0

    scene = FrontendScene(
        center=(center_lng, center_lat),
        default_zoom=14,
        terrain=FrontendTerrain(enabled=True, exaggeration=2.2),
        bbox=(bbox_ll.min_lng, bbox_ll.min_lat, bbox_ll.max_lng, bbox_ll.max_lat),
        layers={
            "field_bbox_geojson": safe_relpath(str(Path("geojson") / "field_bbox.geojson")),
            "severity_geojson": safe_relpath(str(Path("geojson") / "severity_polygons.geojson")),
            "sampling_points_geojson": safe_relpath(str(Path("geojson") / "sampling_points.geojson")),
            "severity_heatmap_png": safe_relpath(str(Path("assets") / "severity_heatmap.png")),
        },
    )

    out = NutrientDeficiencyOutput(
        time_utc=_now_utc_iso(),
        run_id=run_id,
        field_id="field_unknown",
        upstream_path=safe_relpath(str(upstream_path)),
        query_image_path=safe_relpath(str(qpath)),
        decision=DecisionOut(
            is_applicable=applicable,
            upstream_label_conf=float(up_conf),
            agent_validation_score=float(agent_score),
            uncertainty_notes=uncertainty_notes,
        ),
        diagnosis=DiagnosisOut(
            summary_cn=summary_cn,
            summary_en=summary_en,
            suspected_nutrients=suspected,
        ),
        area_stats=AreaStats(
            affected_area_px=affected_area_px,
            affected_ratio=aff_ratio,
            patch_count=patch_count,
            largest_patch_px=largest_patch_px,
        ),
        layers=LayersOut(
            severity_geojson=safe_relpath(str(Path("geojson") / "severity_polygons.geojson")),
            sampling_points_geojson=safe_relpath(str(Path("geojson") / "sampling_points.geojson")),
            severity_heatmap_png=safe_relpath(str(Path("assets") / "severity_heatmap.png")),
            index_preview_png=safe_relpath(str(Path("assets") / "vigor_index.png")),
        ),
        report_md=safe_relpath("report.md"),
        assets={
            "query_preview": q_preview_rel,
            "copied_evidence_count": len(evidence_blocks),
            "evidence_montage": evidence_montage_rel,
            "visuals_triptych": triptych_rel,
            "charts": {
                "low_vigor_hist": safe_relpath(str(Path("assets") / "low_vigor_hist.png")),
                "severity_dist": safe_relpath(str(Path("assets") / "severity_dist.png")),
                "patch_area_bar": safe_relpath(str(Path("assets") / "patch_area_bar.png")),
            },
        },
        frontend_scene=scene,
    )

    out_path = out_dir / "nutrient_deficiency_output.json"
    write_json(out_path, json.loads(out.model_dump_json()))
    write_json(out_dir / "latest.json", json.loads(out.model_dump_json()))
    print(f"[OK] Nutrient agent output written to: {out_dir}")
    print(f" - {out_path}")
    print(f" - {out_dir / 'report.md'}")