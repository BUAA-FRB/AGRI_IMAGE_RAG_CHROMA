from __future__ import annotations

from typing import Any, Dict, List, Optional

from ..io_utils import safe_relpath


def md_image(path: str, alt: str = "") -> str:
    p = safe_relpath(path)
    return f"![{alt}]({p})"


def _md_table(headers: List[str], rows: List[List[str]]) -> str:
    """Simple markdown table generator."""
    if not headers:
        return ""
    line1 = "| " + " | ".join(headers) + " |"
    line2 = "| " + " | ".join(["---"] * len(headers)) + " |"
    body = ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join([line1, line2] + body)


def generate_report_md(
    title_cn: str,
    title_en: str,
    summary_cn: str,
    summary_en: str,
    decision: Dict[str, Any],
    area_stats: Dict[str, Any],
    thresholds: Dict[str, Any],
    images: Dict[str, str],
    evidence_blocks: List[Dict[str, Any]],
    actions_cn: List[str],
    actions_en: List[str],
    monitoring_cn: List[str],
    monitoring_en: List[str],
    extra: Optional[Dict[str, Any]] = None,  # NEW
) -> str:
    extra = extra or {}
    lines: List[str] = []

    lines.append(f"# {title_cn}")
    lines.append("")
    lines.append(f"**{title_en}**")
    lines.append("")

    # -------------------------
    # Summary
    # -------------------------
    lines.append("## 1. 结论摘要 / Summary")
    lines.append("")
    lines.append(f"- **适用性（is_applicable）**：`{decision.get('is_applicable')}`")
    lines.append(f"- **上游置信度（upstream_label_conf）**：`{decision.get('upstream_label_conf'):.3f}`")
    lines.append(f"- **本 Agent 复核分数（agent_validation_score）**：`{decision.get('agent_validation_score'):.3f}`")
    if decision.get("uncertainty_notes"):
        lines.append("- **不确定性说明**：")
        for t in decision["uncertainty_notes"]:
            lines.append(f"  - {t}")
    lines.append("")
    lines.append("**中文：**")
    lines.append("")
    lines.append(summary_cn)
    lines.append("")
    lines.append("**English:**")
    lines.append("")
    lines.append(summary_en)
    lines.append("")

    # -------------------------
    # Visuals (NEW: triptych)
    # -------------------------
    lines.append("## 2. 可视化 / Visuals")
    lines.append("")
    trip = images.get("visuals_triptych")
    if trip:
        lines.append("### Query / Vigor Proxy / Severity Heatmap (1×3)")
        lines.append(md_image(trip, "triptych"))
        lines.append("")
        lines.append("> 从左到右：Query 图像、活力/指数 proxy、严重度热力图（透明 PNG 的可视化版本）。")
        lines.append("")
    else:
        # fallback to old style
        if images.get("query"):
            lines.append("### Query 图像")
            lines.append(md_image(images["query"], "query"))
            lines.append("")
        if images.get("index"):
            lines.append("### 活力/指数图（proxy）")
            lines.append(md_image(images["index"], "index"))
            lines.append("")
        if images.get("heatmap"):
            lines.append("### 严重度热力图（透明 PNG，可叠加）")
            lines.append(md_image(images["heatmap"], "severity_heatmap"))
            lines.append("")

    # -------------------------
    # Extent & severity + table
    # -------------------------
    lines.append("## 3. 范围与严重度 / Extent & Severity")
    lines.append("")
    lines.append(_md_table(
        headers=["Metric", "Value"],
        rows=[
            ["affected_area_px", str(area_stats.get("affected_area_px"))],
            ["affected_ratio", f"{area_stats.get('affected_ratio'):.3f}"],
            ["patch_count", str(area_stats.get("patch_count"))],
            ["largest_patch_px", str(area_stats.get("largest_patch_px"))],
            ["mild_threshold", f"{thresholds.get('mild'):.3f}"],
            ["moderate_threshold", f"{thresholds.get('moderate'):.3f}"],
            ["severe_threshold", f"{thresholds.get('severe'):.3f}"],
        ]
    ))
    lines.append("")

    # -------------------------
    # NEW: Charts section
    # -------------------------
    lines.append("## 4. 更多图表 / More Figures")
    lines.append("")
    if images.get("low_vigor_hist"):
        lines.append("### Low-vigor 直方图（含阈值线）")
        lines.append(md_image(images["low_vigor_hist"], "low_vigor_hist"))
        lines.append("")
    if images.get("severity_dist"):
        lines.append("### Severity 分布（像素计数）")
        lines.append(md_image(images["severity_dist"], "severity_dist"))
        lines.append("")
    if images.get("patch_area_bar"):
        lines.append("### Top Patch 面积柱状图")
        lines.append(md_image(images["patch_area_bar"], "patch_area_bar"))
        lines.append("")

    # severity counts table
    sev_counts = extra.get("severity_counts") or {}
    if sev_counts:
        lines.append("### Severity 分布表")
        rows = [
            ["healthy", str(sev_counts.get("healthy", 0))],
            ["mild", str(sev_counts.get("mild", 0))],
            ["moderate", str(sev_counts.get("moderate", 0))],
            ["severe", str(sev_counts.get("severe", 0))],
        ]
        lines.append(_md_table(["class", "pixel_count"], rows))
        lines.append("")

    # top patch areas table
    top_patches = extra.get("top_patches") or []
    if top_patches:
        lines.append("### Top Patch 面积表（前若干个斑块）")
        rows = []
        for p in top_patches:
            rows.append([str(p.get("rank")), str(p.get("area_px")), str(p.get("bbox_px"))])
        lines.append(_md_table(["rank", "area_px", "bbox(px)"], rows))
        lines.append("")

    # -------------------------
    # Evidence montage
    # -------------------------
    lines.append("## 5. 证据 / Evidence")
    lines.append("")
    montage = images.get("evidence_montage")
    if montage:
        lines.append("### Evidence Montage (2×3)")
        lines.append(md_image(montage, "evidence_montage"))
        lines.append("")
        lines.append("对应关系（从左到右、从上到下）：")
        for blk in evidence_blocks[:6]:
            eid = blk.get("eid", "")
            labels_present = blk.get("labels_present", "")
            note = blk.get("note", "")
            if note:
                lines.append(f"- **{eid}** labels_present={labels_present}；{note}")
            else:
                lines.append(f"- **{eid}** labels_present={labels_present}")
        lines.append("")
    else:
        for blk in evidence_blocks:
            lines.append(f"### {blk.get('eid')}  (labels_present={blk.get('labels_present','')})")
            if blk.get("preview"):
                lines.append(md_image(blk["preview"], blk.get("eid", "")))
            if blk.get("nir_preview"):
                lines.append("")
                lines.append("*NIR preview:*")
                lines.append(md_image(blk["nir_preview"], blk.get("eid", "nir")))
            if blk.get("note"):
                lines.append("")
                lines.append(f"- {blk['note']}")
            lines.append("")

    # -------------------------
    # Actions
    # -------------------------
    lines.append("## 6. 建议动作 / Recommended Actions")
    lines.append("")
    lines.append("**中文：**")
    for a in actions_cn:
        lines.append(f"- {a}")
    lines.append("")
    lines.append("**English:**")
    for a in actions_en:
        lines.append(f"- {a}")
    lines.append("")
    lines.append("> 注意：涉及任何肥料/改良剂/用量方案时，请以当地农艺规范与产品标签为准；如不确定，咨询农技人员。")
    lines.append("")

    # -------------------------
    # Monitoring
    # -------------------------
    lines.append("## 7. 复核与监测 / Verification & Monitoring")
    lines.append("")
    lines.append("**中文：**")
    for m in monitoring_cn:
        lines.append(f"- {m}")
    lines.append("")
    lines.append("**English:**")
    for m in monitoring_en:
        lines.append(f"- {m}")
    lines.append("")

    return "\n".join(lines)