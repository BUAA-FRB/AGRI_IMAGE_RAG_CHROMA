from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .io_utils import relpath_for_md, safe_get_assets_preview_paths
from .models import EconomyResult


# -------------------------
# Formatting helpers
# -------------------------

_FENCE_LINE_RE = re.compile(r"(?m)^\s*```.*\s*$")
_MIXED_NUMBERED_BULLET_RE = re.compile(r"(?m)^\s*[-*+]\s+(\d+)\.\s+")
_HEADING_RE = re.compile(r"(?m)^(#{1,6})\s+")


def _fmt_money(x: Any, currency: str) -> str:
    if x is None:
        return "N/A"
    try:
        v = float(x)
    except Exception:
        return "N/A"
    return f"{currency} {v:,.2f}"


def _fmt_ratio(x: float) -> str:
    return f"{x * 100:.2f}%"


def _strip_outer_markdown_fence(text: str) -> str:
    """
    Remove outermost fenced block if it wraps the whole document, e.g.
    ```markdown
    ...
    ```
    """
    if not text:
        return text
    t = text.strip()

    # If whole text is wrapped
    m = re.match(r"^\s*```(?:markdown|md|text)?\s*\n([\s\S]*?)\n```\s*$", t, flags=re.IGNORECASE)
    if m:
        return m.group(1).strip()

    # If only two fence lines exist and they wrap all
    fence_lines = list(_FENCE_LINE_RE.finditer(t))
    if len(fence_lines) == 2:
        first = fence_lines[0]
        last = fence_lines[1]
        if first.start() == 0 and last.end() == len(t):
            inner = t[first.end(): last.start()]
            return inner.strip()

    return t


def _remove_any_fence_lines(text: str) -> str:
    """
    Remove any standalone fence lines (```xxx), even if nested.
    """
    if not text:
        return text
    return _FENCE_LINE_RE.sub("", text)


def _normalize_mixed_numbered_bullets(text: str) -> str:
    """
    Convert '- 1. xxx' / '* 2. xxx' / '+ 3. xxx' to '1. xxx'
    """
    if not text:
        return text
    return _MIXED_NUMBERED_BULLET_RE.sub(r"\1. ", text)


def _demote_headings_to_h3(text: str) -> str:
    """
    In Section 3 insertion, do not allow # / ##.
    Any heading level (#..######) is demoted to ###.
    """
    if not text:
        return text

    def repl(m: re.Match) -> str:
        # Always demote to ### for stability
        return "### "

    return _HEADING_RE.sub(repl, text)


def _drop_title_or_time_lines(text: str) -> str:
    """
    Drop lines that look like a top-level title or '生成时间' quote to avoid duplication.
    """
    if not text:
        return text
    lines = text.splitlines()
    out: List[str] = []
    for ln in lines:
        s = ln.strip()
        if s.startswith("# "):
            continue
        if "生成时间" in s and s.startswith(">"):
            continue
        out.append(ln)
    return "\n".join(out).strip()


def _sanitize_llm_section(llm_md: Optional[str]) -> str:
    """
    Make LLM output safe to embed:
    - remove outer fence
    - remove any fence lines
    - normalize '- 1.' mixed lists
    - demote headings to ### (no #/##)
    - remove accidental title/time lines
    - trim excessive blank lines
    """
    if not llm_md or not llm_md.strip():
        return ""

    t = llm_md.strip()
    t = _strip_outer_markdown_fence(t)
    t = _remove_any_fence_lines(t)
    t = _normalize_mixed_numbered_bullets(t)
    t = _drop_title_or_time_lines(t)
    t = _demote_headings_to_h3(t)

    # collapse many blank lines
    t = re.sub(r"\n{4,}", "\n\n\n", t).strip()
    return t


# -------------------------
# Report structure helpers
# -------------------------

def _money_unit(econ: EconomyResult) -> str:
    return econ.currency if econ.area_mode == "gsd" else f"{econ.currency}/1{econ.normalize_to}"


def _get_series(charts_meta: Dict[str, Any]) -> Dict[str, Any]:
    return charts_meta.get("series") or {}


def _build_econ_method_note(econ: EconomyResult) -> List[str]:
    """
    A short traceable note to avoid "AI-y" generic talk.
    """
    lines: List[str] = []
    lines.append("- 受灾比例 r 来自 field_stats 的像素面积占比（field_area_px 与 affected_area_px）。")
    lines.append("- 单灾害减产 f(label) 由启发式映射得到（受灾比例→减产），并随置信度扩展不确定性区间（P10/P50/P90）。")
    lines.append("- 总减产使用合并规则避免重复计数：Total = 1 - Π(1 - f_i)（见 assumptions / combine_method）。")
    if econ.area_mode == "gsd":
        lines.append("- 金额口径：BaseFieldValue（地块总产值）× TotalYieldLossFrac（总减产比例）得到总损失。")
    else:
        lines.append("- 金额口径：当前为标准化损失（每 1 亩/公顷），用于在未知真实面积时做横向比较。")
    return lines


def _build_economic_actions(econ: EconomyResult, charts_meta: Dict[str, Any]) -> List[str]:
    """
    Structured, economy-oriented actions. Keep it not overly agronomic.
    """
    series = _get_series(charts_meta)
    red = float(series.get("mitigation_reduction") or 0.0)
    red_pct = max(0.0, min(0.95, red)) * 100.0

    labels = {it.label for it in econ.items}
    hints: List[str] = []
    hints.append(f"- 情景推演显示：若及时处置，最终损失可能降低约 {red_pct:.1f}%（来自趋势推演参数）。")

    # P0: low-cost/high-return generic moves, hazard-informed
    p0: List[str] = []
    p0.append("建立“损失台账最小集”：地块ID/时间/影像证据编号/受灾类型/初判范围/处置动作/责任人/复测时间。")
    p0.append("优先做“止损可逆项”的核验：本次损失是否来自可短期纠正的因素（如排水/通道异常/局部倒伏），避免把可止损损失当作既定损失。")
    if "waterway" in labels or "water" in labels:
        p0.append("对水相关风险：先确认是否存在可快速恢复的关键节点（堵点/倒灌/低洼积水），并记录‘持续时长’以区分轻微影响与减产显著情形。")
    if "storm_damage" in labels:
        p0.append("对风暴损伤：记录受损等级与可恢复性（倒伏比例、折断比例），用于把“暂时性影响”与“不可逆减产”分开核算。")
    p0.append("用少量样方/小区抽样给出‘程度系数’（轻/中/重），把面积口径升级为“面积×程度”的经济口径。")

    # P1: verification + ROI
    p1: List[str] = []
    p1.append("把关键参数做成‘三点估计’：面积（或GSD）、价格/单产、灾害持续/强度；每项给 P10/P50/P90，形成可解释的不确定性来源。")
    p1.append("对高占比灾害做一次“处置 ROI 评估”：投入（人工/机械/材料）与可挽回损失对比，优先投入边际收益最高的动作。")
    p1.append("将影像证据与现场数据对齐：同一地块同一时间段的实测（含照片编号）与模型输出建立一一映射，便于审计与复盘。")

    # P2: systematization
    p2: List[str] = []
    p2.append("建立长期经济监测：按作物/地块沉淀单产与价格的历史分布，用于替换占位参数。")
    p2.append("建立阈值化告警：受灾比例、持续时长、复测变化率达到阈值才触发更高成本处置，避免过度投入。")
    p2.append("复盘：对照收获期实际产量，校准‘受灾比例→减产’映射，形成本地化系数库。")

    out: List[str] = []
    out.append("### 7.1 分阶段止损建议（经济视角）")
    out.append("> 优先级：P0 24–72h｜P1 1–2周｜P2 长期")
    out.append("")
    out.append("**P0（24–72h）**")
    for x in p0:
        out.append(f"- {x}")
    out.append("")
    out.append("**P1（1–2周）**")
    for x in p1:
        out.append(f"- {x}")
    out.append("")
    out.append("**P2（长期）**")
    for x in p2:
        out.append(f"- {x}")
    out.append("")
    out.append("### 7.2 情景解释（为什么这些动作能止损）")
    for x in hints:
        out.append(x)
    out.append("- 机制：通过缩短灾害持续、降低不可逆损伤比例、提升处置命中率，把“潜在损失”转化为“可挽回损失”。")
    return out


def build_economy_report_markdown(
    latest: Dict[str, Any],
    econ: EconomyResult,
    charts_meta: Dict[str, Any],
    out_dir: Path,
    llm_md: Optional[str] = None,
) -> str:
    """
    Multimodal Markdown report with strong formatting stability.
    LLM output is sanitized and embedded ONLY inside Section 3.
    """
    now_utc = datetime.now(timezone.utc).isoformat()

    normalized = latest.get("normalized") or {}
    conclusion = normalized.get("conclusion") or (latest.get("parsed_json") or {}).get("conclusion") or ""
    uncertainty = normalized.get("uncertainty") or (latest.get("parsed_json") or {}).get("uncertainty") or ""

    # chart paths (relative to out_dir)
    charts_dir = Path(charts_meta["charts_dir"])
    loss_forecast_png = Path(charts_meta["loss_forecast_png"])
    breakdown_bar_png = Path(charts_meta["breakdown_bar_png"])
    breakdown_pie_png = Path(charts_meta["breakdown_pie_png"])
    charts_data_json = Path(charts_meta["charts_data_json"])

    loss_forecast_md = relpath_for_md(loss_forecast_png, out_dir)
    breakdown_bar_md = relpath_for_md(breakdown_bar_png, out_dir)
    breakdown_pie_md = relpath_for_md(breakdown_pie_png, out_dir)
    charts_data_md = relpath_for_md(charts_data_json, out_dir)

    # previews (multimodal images)
    previews = safe_get_assets_preview_paths(latest)
    query_preview = previews.get("query_preview")
    ev_preview = previews.get("evidence_preview")
    ev_nir_preview = previews.get("evidence_nir_preview")

    output_dir = latest.get("assets", {}).get("output_dir")
    if output_dir:
        base = Path(output_dir)
        q_md = relpath_for_md(base / query_preview, out_dir) if query_preview else None
        e_md = relpath_for_md(base / ev_preview, out_dir) if ev_preview else None
        en_md = relpath_for_md(base / ev_nir_preview, out_dir) if ev_nir_preview else None
    else:
        q_md = f"../{query_preview}" if query_preview else None
        e_md = f"../{ev_preview}" if ev_preview else None
        en_md = f"../{ev_nir_preview}" if ev_nir_preview else None

    # forecast table
    month_labels = charts_meta.get("month_labels") or []
    series = _get_series(charts_meta)
    base_p50 = (series.get("baseline") or {}).get("p50") or []
    base_p10 = (series.get("baseline") or {}).get("p10") or []
    base_p90 = (series.get("baseline") or {}).get("p90") or []
    mitigation_reduction = float(series.get("mitigation_reduction") or 0.0)

    unit = _money_unit(econ)

    # sanitize llm section
    llm_section = _sanitize_llm_section(llm_md)

    lines: List[str] = []
    lines.append("# 智能农业经济损失估计与趋势推演报告")
    lines.append(f"> 生成时间：{now_utc}")
    lines.append("")

    # 1) One-page summary
    lines.append("## 1. 关键信息概览（一页摘要）")
    lines.append(f"- 核心结论：{conclusion}".strip())
    lines.append(f"- 不确定性：{uncertainty}".strip())

    labels = [it.label for it in econ.items]
    lines.append(f"- 主要灾害标签：{', '.join(labels) if labels else '无'}")
    lines.append(f"- 作物假设：{econ.crop}（stage={econ.stage}）")
    if econ.area_mode == "gsd":
        lines.append(f"- 面积口径：gsd={econ.gsd_m_per_px} m/px（field: gsd_m_per_px），金额单位：{unit}")
    else:
        lines.append(f"- 面积口径：ratio_only（归一化到 1{econ.normalize_to}），金额单位：{unit}")

    lines.append(
        f"- 预估总损失：P10={_fmt_money(econ.total_loss_value_p10, econ.currency)}，"
        f"P50={_fmt_money(econ.total_loss_value_p50, econ.currency)}，"
        f"P90={_fmt_money(econ.total_loss_value_p90, econ.currency)}"
        + (f"（单位：{unit}）" if econ.area_mode == "ratio_only" else "")
    )
    lines.append(
        f"- 预估总减产比例：P10={_fmt_ratio(econ.total_yield_loss_frac_p10)}，"
        f"P50={_fmt_ratio(econ.total_yield_loss_frac_p50)}，"
        f"P90={_fmt_ratio(econ.total_yield_loss_frac_p90)}"
    )
    lines.append(
        f"- 处置情景止损空间：最终损失可能降低约 {mitigation_reduction * 100:.1f}%（来自趋势推演参数）"
    )
    lines.append("")

    # 1.5 charts
    lines.append("## 1.5 图表与量化推演（用于辅助理解，可替换为真实监测数据）")
    lines.append("> 说明：以下图表由规则经济映射 + 不确定性区间 + 情景推演生成，用于把“受灾面积”转换为“损失规模与节奏”。不替代实地抽样、产量统计与价格台账。")
    lines.append("")

    lines.append("### 1.5.1 未来数月经济损失走向（累计口径）")
    lines.append("带状范围表示不确定性区间（P10–P90）。同时给出“处置后情景”曲线，用于量化止损空间。")
    lines.append("")
    lines.append(f"![]({loss_forecast_md})")
    lines.append("")
    lines.append("**未来数月累计损失（基线P50，单位同上）**")
    lines.append("| 月份 | P50(基线) | 区间(P10–P90) |")
    lines.append("|---|---:|---:|")
    for i in range(min(len(month_labels), len(base_p50), len(base_p10), len(base_p90))):
        lines.append(f"| {month_labels[i]} | {base_p50[i]:,.2f} | {base_p10[i]:,.2f}–{base_p90[i]:,.2f} |")
    lines.append("")

    lines.append("### 1.5.2 损失归因构成（按灾害标签）")
    lines.append("用于快速判断“主要损失来自哪里”，并据此确定排查与投入优先级。")
    lines.append("")
    lines.append(f"![]({breakdown_bar_md})")
    lines.append("")
    lines.append(f"![]({breakdown_pie_md})")
    lines.append("")
    lines.append(f"> 图表数据已保存：`{charts_data_md}`（后续可用真实监测数据替换启发式推演）。")
    lines.append("")

    # 2) Evidence images (multimodal)
    lines.append("## 2. 影像证据（多模态展示）")
    lines.append("> 为了让经济估算“有迹可循”，此处展示 query 与检索到的最相似证据样本（RGB/NIR 预览图，如存在）。")
    lines.append("")
    if q_md:
        lines.append("**Query 预览：**")
        lines.append("")
        lines.append(f"![]({q_md})")
        lines.append("")
    if e_md:
        lines.append("**Best Evidence（RGB）预览：**")
        lines.append("")
        lines.append(f"![]({e_md})")
        lines.append("")
    if en_md:
        lines.append("**Best Evidence（NIR）预览：**")
        lines.append("")
        lines.append(f"![]({en_md})")
        lines.append("")

    # 3) LLM analysis (sanitized)
    lines.append("## 3. 经济研判结论（大模型分析）")
    if llm_section:
        lines.append(llm_section)
    else:
        lines.append("> 未启用大模型分析或输出为空（可用 `--no-llm` 仅生成规则报告与图表）。")
    lines.append("")

    # 4) Detailed breakdown table
    lines.append("## 4. 分项损失明细（结构化）")
    lines.append("")
    lines.append("| label | conf | 受灾比例 | 减产P10/P50/P90 | 损失P10/P50/P90 |")
    lines.append("|---|---:|---:|---:|---:|")
    for it in econ.items:
        yl = f"{_fmt_ratio(it.yield_loss_frac_p10)}/{_fmt_ratio(it.yield_loss_frac_p50)}/{_fmt_ratio(it.yield_loss_frac_p90)}"
        lv = f"{_fmt_money(it.loss_value_p10, econ.currency)}/{_fmt_money(it.loss_value_p50, econ.currency)}/{_fmt_money(it.loss_value_p90, econ.currency)}"
        lines.append(f"| {it.label} | {it.confidence:.2f} | {_fmt_ratio(it.affected_ratio)} | {yl} | {lv} |")
    lines.append("")

    # 5) Method / traceability
    lines.append("## 5. 口径与计算链（可追溯）")
    lines.append("- 目标：把“受灾范围”转换为“减产比例”和“经济损失”，并给出 P10/P50/P90 不确定性区间。")
    for x in _build_econ_method_note(econ):
        lines.append(x)
    if econ.base_field_value_p50 is not None:
        lines.append(f"- BaseFieldValue（P50）：{_fmt_money(econ.base_field_value_p50, econ.currency)}（field: base_field_value_p50，单位口径：{unit}）")
    if econ.assumptions:
        lines.append("- assumptions（摘要）：")
        # keep it short and readable
        for k in ("area_mode", "gsd_m_per_px", "normalize_to", "combine_method", "tile_id_resolution", "crop_params"):
            if k in econ.assumptions:
                lines.append(f"  - {k}: {econ.assumptions[k]}")
    lines.append("")

    # 6) Uncertainty & sensitivity
    lines.append("## 6. 风险与不确定性（为什么区间会变宽/怎么变窄）")
    lines.append("- **面积换算敏感**：若 `area_mode=gsd`，损失与 `gsd_m_per_px` 的平方成正比；若暂无法确认真实 GSD，建议先用 `ratio_only` 输出标准化损失。")
    lines.append("- **价格/单产敏感**：当前作物经济参数可能为占位初值，建议替换为地区台账或统计口径。")
    lines.append("- **强度/持续时间缺失**：仅靠面积可能高估或低估，建议补充灾害持续天数/等级，把模型从“面积口径”升级为“面积×程度”。")
    lines.append("- **tile 代理风险**：若 query 不是标准 tile，而使用最相似证据 tile 作为 proxy，则需要现场核验其一致性（见 assumptions.tile_id_resolution）。")
    if econ.warnings:
        lines.append("")
        lines.append("**降级/警告：**")
        for w in econ.warnings:
            lines.append(f"- {w}")
    lines.append("")

    # 7) Structured economic actions
    lines.append("## 7. 经济处置策略（止损导向，结构化）")
    lines.extend(_build_economic_actions(econ, charts_meta))
    lines.append("")

    # 8) Data checklist / ledger template
    lines.append("## 8. 仍需补充的信息与经济台账模板（可直接抄用）")
    lines.append("### 8.1 优先补充信息（能显著提高估计质量）")
    lines.append("- 真实地块面积（或真实 GSD / GeoTIFF 仿射变换），用于把像素面积映射到亩/公顷。")
    lines.append("- 作物类型与生育期（至少到地块级/批次级），用于选择更准确的减产系数。")
    lines.append("- 本地价格与单产口径（品种/等级/收购价、历史均值与波动区间）。")
    lines.append("- 灾害持续时长/强度（积水天数、旱情等级、风暴等级等），用于从“面积”升级到“程度”。")
    lines.append("- 处置投入清单（人工/机械/材料/时长），用于做 ROI（投入-挽回损失）评估。")
    lines.append("")
    lines.append("### 8.2 经济取证与台账字段（建议最小集）")
    lines.append("| 字段 | 示例 | 说明 |")
    lines.append("|---|---|---|")
    lines.append("| 地块ID | tile_id / 自定义地块号 | 与影像/实测一一对应 |")
    lines.append("| 时间 | YYYY-MM-DD hh:mm | 便于与影像时间对齐 |")
    lines.append("| 灾害类型 | waterway / water 等 | 与预测标签一致 |")
    lines.append("| 初判范围 | 受灾比例/面积 | 优先记录比例，面积待补 GSD |")
    lines.append("| 程度等级 | 轻/中/重 | 面积×程度口径的关键 |")
    lines.append("| 处置动作 | 排水/修复/补施等 | 只记动作，不必写农技长文 |")
    lines.append("| 投入成本 | 金额/工时/机械时长 | 用于 ROI |")
    lines.append("| 复测结果 | 范围变化/程度变化 | 用于验证止损效果 |")
    lines.append("| 证据编号 | 照片/视频/影像路径 | 可追溯审计 |")
    lines.append("| 责任人 | 姓名/班组 | 管理闭环 |")
    lines.append("")

    # 9) attachments
    lines.append("## 9. 附件与可追溯文件")
    lines.append(f"- 结构化输出：`economy_output.json`（在 {out_dir.as_posix()} 下）")
    lines.append(f"- 图表目录：`{relpath_for_md(charts_dir, out_dir)}`")
    lines.append("- 若启用大模型：`llm_analysis.md` 与 `economy_prompt.txt`（便于复现与审计）")
    lines.append("")

    return "\n".join(lines).strip() + "\n"
