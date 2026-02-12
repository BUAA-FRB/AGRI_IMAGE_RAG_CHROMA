from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import math
import json


# -------------------------
# Global chart style (consistent size)
# -------------------------

FIGSIZE = (8.0, 4.8)   # inches -> fixed output size
DPI = 180              # fixed dpi -> fixed pixels
LINEWIDTH = 2.2


@dataclass
class ChartBundle:
    run_id: str
    charts_dir: Path
    use_cjk: bool
    font_note: str

    severity_png: Optional[Path] = None
    hazard_pie_png: Optional[Path] = None
    actions_bar_png: Optional[Path] = None
    data_json: Optional[Path] = None

    # data for markdown tables
    dates: Optional[List[str]] = None
    severity: Optional[List[float]] = None
    severity_lo: Optional[List[float]] = None
    severity_hi: Optional[List[float]] = None
    hazard_probs: Optional[Dict[str, float]] = None
    action_counts: Optional[Dict[str, int]] = None


def _risk_to_base(risk_level: str) -> float:
    r = (risk_level or "").strip().lower()
    if r in ("high", "h", "严重", "高", "高风险"):
        return 75.0
    if r in ("medium", "m", "中等", "中", "中风险"):
        return 55.0
    if r in ("low", "l", "轻微", "低", "低风险"):
        return 35.0
    return 50.0


def _conf01(confidence: float) -> float:
    c = confidence
    if c > 1.0:
        c = c / 100.0
    return max(0.0, min(1.0, c))


def _compute_severity_series(
    start_time_utc: str,
    base: float,
    confidence: float,
    days: int = 14,
) -> Tuple[List[str], List[float], List[float], List[float]]:
    """
    Heuristic severity forecast (0-100).
    - assume mitigation actions gradually take effect after day 2
    - higher confidence => narrower uncertainty band
    """
    try:
        dt0 = datetime.fromisoformat(start_time_utc.replace("Z", "+00:00"))
    except Exception:
        dt0 = datetime.now(timezone.utc)

    conf = _conf01(confidence)
    band = 18.0 - 10.0 * conf  # 8..18 approx

    dates: List[str] = []
    sev: List[float] = []
    lo: List[float] = []
    hi: List[float] = []

    for t in range(days):
        day = dt0 + timedelta(days=t)
        dates.append(day.strftime("%Y-%m-%d"))

        bump = 4.0 * math.exp(-((t - 1.0) ** 2) / 2.0)  # max around day 1
        decay = 18.0 * (1.0 - math.exp(-max(0.0, t - 2.0) / 4.0))
        value = base + bump - decay

        value = max(0.0, min(100.0, value))
        sev.append(value)
        lo.append(max(0.0, value - band))
        hi.append(min(100.0, value + band))

    return dates, sev, lo, hi


def _hazard_probabilities_from_label(canon_label: str) -> Dict[str, float]:
    """
    Heuristic hazard distribution (pie). Not a real probabilistic model.
    """
    key = (canon_label or "").strip().lower()

    mapping: Dict[str, Dict[str, float]] = {
        "water_way": {
            "沟渠/水道异常": 0.46,
            "田间积水/渍害": 0.18,
            "失墒/干旱": 0.14,
            "设施渗漏/损坏": 0.12,
            "其他/待核查": 0.10,
        },
        "water": {
            "田间积水/渍害": 0.48,
            "沟渠/水道异常": 0.20,
            "病害诱发风险": 0.14,
            "土壤缺氧风险": 0.10,
            "其他/待核查": 0.08,
        },
        "drydown": {
            "失墒/干旱": 0.50,
            "灌溉不足": 0.18,
            "高温蒸散": 0.12,
            "土壤板结/保墒不足": 0.12,
            "其他/待核查": 0.08,
        },
        "nutrient_deficiency": {
            "营养缺乏": 0.52,
            "水分胁迫": 0.16,
            "病虫害/杂草干扰": 0.14,
            "土壤问题": 0.10,
            "其他/待核查": 0.08,
        },
    }

    if key in mapping:
        return mapping[key]

    return {
        f"{canon_label or '主要风险'}": 0.45,
        "水分异常相关": 0.18,
        "植株生理胁迫": 0.15,
        "设施/管理问题": 0.12,
        "其他/待核查": 0.10,
    }


def _count_actions_by_phase(phases: Dict[str, list]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for name in ["灾前", "灾中", "灾后"]:
        arr = phases.get(name, []) if isinstance(phases, dict) else []
        counts[name] = len(arr) if isinstance(arr, list) else 0
    return counts


def _ensure_matplotlib_and_fonts() -> Tuple[object, object, bool, str]:
    """
    Returns (plt, np, use_cjk, font_note).
    - If no CJK fonts are available, we fall back to English labels.
    """
    import matplotlib
    matplotlib.use("Agg")  # headless safe
    import matplotlib.pyplot as plt  # noqa
    import numpy as np  # noqa
    from matplotlib import font_manager as fm  # noqa

    # prefer common CJK fonts (Windows & Linux & macOS)
    candidates = [
        "Microsoft YaHei",
        "SimHei",
        "PingFang SC",
        "Hiragino Sans GB",
        "Noto Sans CJK SC",
        "Noto Sans CJK JP",
        "WenQuanYi Micro Hei",
        "Source Han Sans SC",
        "Arial Unicode MS",
    ]

    available_names = set()
    try:
        for f in fm.fontManager.ttflist:
            if getattr(f, "name", None):
                available_names.add(f.name)
    except Exception:
        available_names = set()

    found = [name for name in candidates if name in available_names]
    use_cjk = len(found) > 0

    if use_cjk:
        plt.rcParams["font.sans-serif"] = found[:2] + ["DejaVu Sans"]
        font_note = ""
    else:
        # fallback to English-only to avoid tofu/legend issues
        plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
        font_note = "检测到环境缺少可用中文字体：图表将自动使用英文标注以避免乱码。可安装/配置中文字体后重跑生成中文图表。"

    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.facecolor"] = "white"
    plt.rcParams["savefig.facecolor"] = "white"
    plt.rcParams["axes.grid"] = True
    plt.rcParams["grid.alpha"] = 0.25
    plt.rcParams["grid.linestyle"] = "-"
    plt.rcParams["axes.titlepad"] = 10

    return plt, np, use_cjk, font_note


def _chart_paths(charts_dir: Path) -> Tuple[Path, Path, Path, Path]:
    p1 = charts_dir / "severity_forecast.png"
    p2 = charts_dir / "hazard_prob_pie.png"
    p3 = charts_dir / "actions_by_phase.png"
    pj = charts_dir / "charts_data.json"
    return p1, p2, p3, pj


def generate_charts(
    out_dir_p: Path,
    run_id: str,
    time_utc: str,
    canon_label: str,
    risk_level: str,
    confidence: float,
    phases: Dict[str, list],
    days: int = 14,
) -> ChartBundle:
    """
    Generate charts into out_dir/assets/charts/<run_id>/ and return ChartBundle.
    Guarantees consistent figure size across charts.
    """
    charts_dir = out_dir_p / "assets" / "charts" / run_id
    charts_dir.mkdir(parents=True, exist_ok=True)

    plt, np, use_cjk, font_note = _ensure_matplotlib_and_fonts()
    bundle = ChartBundle(run_id=run_id, charts_dir=charts_dir, use_cjk=use_cjk, font_note=font_note)

    p1, p2, p3, pj = _chart_paths(charts_dir)

    # data
    base = _risk_to_base(risk_level)
    dates, sev, lo, hi = _compute_severity_series(time_utc, base, confidence, days=days)
    bundle.dates, bundle.severity, bundle.severity_lo, bundle.severity_hi = dates, sev, lo, hi

    probs = _hazard_probabilities_from_label(canon_label)
    bundle.hazard_probs = probs

    action_counts = _count_actions_by_phase(phases)
    bundle.action_counts = action_counts

    # labels: if no CJK, switch to English to avoid font missing
    if use_cjk:
        xlab = "日期"
        ylab = "严重程度指数（0-100）"
        t_line = "未来一段时间灾情严重程度变化（启发式推演）"
        t_pie = "可能相关灾害类型概率分布（启发式）"
        t_bar = "分阶段行动建议数量概览"
        xlab_bar = "阶段"
        ylab_bar = "建议条目数"
    else:
        xlab = "Date"
        ylab = "Severity Index (0-100)"
        t_line = "Severity Trend (Heuristic)"
        t_pie = "Hazard Type Probabilities (Heuristic)"
        t_bar = "Action Count by Phase"
        xlab_bar = "Phase"
        ylab_bar = "Count"

        # also translate hazard labels to avoid tofu
        # keep values the same
        probs_en: Dict[str, float] = {}
        for k, v in probs.items():
            # simple safe fallback: keep ASCII-only if possible
            probs_en[k.encode("ascii", "ignore").decode("ascii") or "Category"] = v
        probs = probs_en
        bundle.hazard_probs = probs

        action_counts_en = {"Pre": action_counts.get("灾前", 0), "During": action_counts.get("灾中", 0), "Post": action_counts.get("灾后", 0)}
        action_counts = action_counts_en
        bundle.action_counts = action_counts

    # -------------------------
    # 1) line chart (fixed size)
    # -------------------------
    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)
    x = np.arange(len(dates))
    ax.plot(x, sev, linewidth=LINEWIDTH)
    ax.fill_between(x, lo, hi, alpha=0.18)

    tick_step = max(1, len(dates) // 7)
    xticks = x[::tick_step]
    xlabels = [dates[i] for i in range(0, len(dates), tick_step)]
    ax.set_xticks(xticks)
    ax.set_xticklabels(xlabels, rotation=25, ha="right")

    ax.set_ylim(0, 100)
    ax.set_xlim(-0.5, len(dates) - 0.5)
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    ax.set_title(t_line)
    ax.grid(True, alpha=0.25)

    fig.subplots_adjust(left=0.10, right=0.98, bottom=0.22, top=0.88)
    fig.savefig(p1, dpi=DPI)
    plt.close(fig)
    bundle.severity_png = p1

    # -------------------------
    # 2) pie chart (fixed size)
    # -------------------------
    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)
    labels = list(probs.keys())
    values = list(probs.values())

    # keep label sizes readable; avoid too tight layout issues
    ax.pie(
        values,
        labels=labels,
        autopct="%1.0f%%",
        pctdistance=0.72,
        labeldistance=1.05,
        textprops={"fontsize": 10},
    )
    ax.set_title(t_pie)
    ax.axis("equal")
    fig.subplots_adjust(left=0.04, right=0.96, bottom=0.08, top=0.88)
    fig.savefig(p2, dpi=DPI)
    plt.close(fig)
    bundle.hazard_pie_png = p2

    # -------------------------
    # 3) bar chart (fixed size)
    # -------------------------
    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)
    ph = list(action_counts.keys())
    ct = [action_counts[k] for k in ph]
    ax.bar(ph, ct)
    ax.set_xlabel(xlab_bar)
    ax.set_ylabel(ylab_bar)
    ax.set_title(t_bar)
    ax.grid(True, axis="y", alpha=0.25)
    fig.subplots_adjust(left=0.10, right=0.98, bottom=0.18, top=0.88)
    fig.savefig(p3, dpi=DPI)
    plt.close(fig)
    bundle.actions_bar_png = p3

    # persist data json
    data = {
        "meta": {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "note": "启发式推演数据：用于报告可视化展示，不替代真实监测/气象/农情数据。",
            "font_note": font_note,
        },
        "severity_forecast": {
            "dates": dates,
            "severity": sev,
            "severity_lo": lo,
            "severity_hi": hi,
        },
        "hazard_probabilities": probs,
        "actions_by_phase": action_counts,
    }
    pj.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    bundle.data_json = pj

    return bundle


def build_charts_markdown_section(bundle: ChartBundle) -> str:
    """
    Return a markdown section that embeds the generated charts.
    Paths are relative to report_out (out_dir).
    """
    def rel(p: Path) -> str:
        return f"assets/charts/{bundle.run_id}/{p.name}".replace("\\", "/")

    lines: List[str] = []
    lines.append("## 1.5 图表与量化推演（用于辅助理解，可替换为真实监测数据）")
    lines.append("> 说明：以下图表由规则/启发式推演生成，用来帮助把“风险”转换成“节奏”和“优先级”。不替代气象、墒情与现场踏查。")
    if bundle.font_note:
        lines.append(f"> 字体提示：{bundle.font_note}")
    lines.append("")

    # 1) severity
    if bundle.severity_png:
        lines.append("### 1.5.1 严重程度变化趋势（折线图）")
        lines.append("可以把它理解为“按建议处置后的风险走向”。带状范围表示不确定性区间。")
        lines.append("")
        lines.append(f"![]({rel(bundle.severity_png)})")
        lines.append("")

        if bundle.dates and bundle.severity and bundle.severity_lo and bundle.severity_hi:
            lines.append("**近 7 日指数（0-100）**（便于写入台账/周报）")
            lines.append("| 日期 | 指数 | 区间(低-高) |")
            lines.append("|---|---:|---:|")
            for i in range(min(7, len(bundle.dates))):
                lines.append(
                    f"| {bundle.dates[i]} | {bundle.severity[i]:.1f} | {bundle.severity_lo[i]:.1f}–{bundle.severity_hi[i]:.1f} |"
                )
            lines.append("")

    lines.append("---")
    lines.append("")

    # 2) pie
    if bundle.hazard_pie_png:
        lines.append("### 1.5.2 相关风险类型构成（饼图）")
        lines.append("这不是“事实统计”，而是基于当前标签与知识条目做的可视化归纳，方便决定先排查哪一类。")
        lines.append("")
        lines.append(f"![]({rel(bundle.hazard_pie_png)})")
        lines.append("")

    lines.append("---")
    lines.append("")

    # 3) bar
    if bundle.actions_bar_png:
        lines.append("### 1.5.3 行动建议的阶段分布（柱状图）")
        lines.append("用于快速判断工作量主要集中在哪个阶段，避免把所有事项挤到同一天。")
        lines.append("")
        lines.append(f"![]({rel(bundle.actions_bar_png)})")
        lines.append("")

    if bundle.data_json:
        lines.append(f"> 图表数据已保存：`{rel(bundle.data_json)}`（后续可用真实监测数据替换这些启发式数据）。")

    lines.append("")
    return "\n".join(lines)
