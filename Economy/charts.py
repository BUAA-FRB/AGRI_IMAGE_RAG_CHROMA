from __future__ import annotations

import json
import math
import re
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt

from .models import EconomyResult


# -------------------------
# Fonts & basic style
# -------------------------

def _configure_chinese_fonts() -> None:
    """
    Try to enable Chinese characters in matplotlib.
    """
    try:
        plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
        plt.rcParams["axes.unicode_minus"] = False
    except Exception:
        pass


def _apply_readable_style() -> None:
    """
    Make charts look cleaner without hard-coding specific colors.
    """
    try:
        plt.rcParams["axes.grid"] = True
        plt.rcParams["grid.alpha"] = 0.25
        plt.rcParams["grid.linestyle"] = "--"
        plt.rcParams["axes.axisbelow"] = True
        plt.rcParams["figure.dpi"] = 120
        plt.rcParams["savefig.dpi"] = 220
    except Exception:
        pass


# -------------------------
# Display helpers
# -------------------------

_LABEL_CN = {
    "planter_skip": "漏播/缺苗",
    "water": "积水",
    "weed_cluster": "杂草聚集",
    "drydown": "干旱/干燥胁迫",
    "waterway": "水道/沟渠异常",
    "storm_damage": "风暴损伤",
    "double_plant": "重播/重叠播种",
    "nutrient_deficiency": "营养缺乏",
    "endrow": "地头行异常",
}


def _label_display(label: str) -> str:
    cn = _LABEL_CN.get(label, "")
    if cn:
        return f"{cn}（{label}）"
    return label


def _money_unit(res: EconomyResult) -> str:
    if res.area_mode == "ratio_only":
        return f"{res.currency} / 1{res.normalize_to}"
    return res.currency


def _month_labels(n_months: int, start: Optional[datetime] = None) -> List[str]:
    start = start or datetime.now()
    y, m = start.year, start.month
    labels = []
    for i in range(n_months + 1):
        mm = m + i
        yy = y + (mm - 1) // 12
        mo = ((mm - 1) % 12) + 1
        labels.append(f"{yy:04d}-{mo:02d}")
    return labels


# -------------------------
# Hazard heuristics (time & mitigation)
# -------------------------

def _hazard_tau_months(label: str) -> float:
    table = {
        "storm_damage": 0.8,
        "water": 1.0,
        "waterway": 1.2,
        "drydown": 1.8,
        "planter_skip": 1.5,
        "weed_cluster": 2.2,
        "nutrient_deficiency": 2.6,
        "double_plant": 1.4,
        "endrow": 2.8,
    }
    return float(table.get(label, 2.0))


def _mitigation_reduction(label: str) -> float:
    table = {
        "waterway": 0.30,
        "water": 0.25,
        "storm_damage": 0.10,
        "drydown": 0.20,
        "nutrient_deficiency": 0.25,
        "weed_cluster": 0.20,
        "planter_skip": 0.10,
        "double_plant": 0.05,
        "endrow": 0.05,
    }
    return float(table.get(label, 0.15))


# -------------------------
# Single-label refinement (components & drivers)
# -------------------------

_COMPONENT_WEIGHTS_BY_LABEL: Dict[str, List[Tuple[str, float]]] = {
    # 经济口径：减产 / 品质 / 处置成本（比例为启发式，用于“解释结构”，不是事实统计）
    "waterway": [("减产损失", 0.70), ("品质/等级损失", 0.15), ("追加处置成本", 0.15)],
    "water": [("减产损失", 0.75), ("品质/等级损失", 0.10), ("追加处置成本", 0.15)],
    "drydown": [("减产损失", 0.80), ("品质/等级损失", 0.15), ("追加处置成本", 0.05)],
    "storm_damage": [("减产损失", 0.85), ("品质/等级损失", 0.10), ("追加处置成本", 0.05)],
    "nutrient_deficiency": [("减产损失", 0.60), ("品质/等级损失", 0.25), ("追加处置成本", 0.15)],
    "weed_cluster": [("减产损失", 0.55), ("品质/等级损失", 0.20), ("追加处置成本", 0.25)],
    "planter_skip": [("减产损失", 0.65), ("品质/等级损失", 0.05), ("追加处置成本", 0.30)],
    "double_plant": [("减产损失", 0.35), ("品质/等级损失", 0.15), ("追加处置成本", 0.50)],
    "endrow": [("减产损失", 0.45), ("品质/等级损失", 0.20), ("追加处置成本", 0.35)],
}


_DRIVER_TEMPLATES: Dict[str, List[Tuple[str, float, List[str]]]] = {
    # (driver_name, base_weight, keywords)
    "waterway": [
        ("沟渠堵塞/淤积", 0.45, ["堵塞", "淤积", "清淤", "杂物", "堵点"]),
        ("排水不畅/水位异常", 0.35, ["水位", "倒灌", "排水", "积水", "水位异常"]),
        ("渠段破损/渗漏", 0.20, ["破损", "渗漏", "塌陷", "损坏", "裂缝"]),
    ],
    "water": [
        ("短时积水（可排）", 0.55, ["短时", "排涝", "可排"]),
        ("持续积水（缺氧）", 0.30, ["持续", "缺氧", "烂根"]),
        ("低洼/排水系统不足", 0.15, ["低洼", "排水系统", "沟渠不足"]),
    ],
    "drydown": [
        ("墒情不足/持续少雨", 0.55, ["少雨", "干旱", "墒情"]),
        ("灌溉能力不足", 0.30, ["灌溉", "水源不足"]),
        ("高温蒸散偏强", 0.15, ["高温", "蒸散"]),
    ],
}


def _normalize_weights(items: List[Tuple[str, float]]) -> List[Tuple[str, float]]:
    s = sum(max(0.0, w) for _, w in items)
    if s <= 1e-12:
        n = len(items) if items else 1
        return [(name, 1.0 / n) for name, _ in items] if items else [("未知", 1.0)]
    return [(name, max(0.0, w) / s) for name, w in items]


def _extract_context_text(latest: Optional[Dict[str, Any]]) -> str:
    if not isinstance(latest, dict):
        return ""
    normalized = latest.get("normalized") or {}
    conclusion = normalized.get("conclusion") or (latest.get("parsed_json") or {}).get("conclusion") or ""
    uncertainty = normalized.get("uncertainty") or (latest.get("parsed_json") or {}).get("uncertainty") or ""
    # also include label reasons if any
    preds = normalized.get("predicted_labels") or (latest.get("parsed_json") or {}).get("predicted_labels") or []
    reasons = []
    if isinstance(preds, list):
        for p in preds[:3]:
            if isinstance(p, dict):
                r = p.get("reason") or ""
                if isinstance(r, str) and r.strip():
                    reasons.append(r.strip())
    parts = [conclusion.strip(), uncertainty.strip(), " ".join(reasons).strip()]
    return " ".join([p for p in parts if p]).strip()


def _infer_drivers(label: str, context_text: str) -> Tuple[List[Tuple[str, float]], List[str]]:
    """
    Return (drivers, basis_keywords_used)
    drivers: list of (driver_name, weight)
    """
    tpl = _DRIVER_TEMPLATES.get(label)
    if not tpl:
        return [], []

    text = context_text or ""
    used: List[str] = []
    boosted: List[Tuple[str, float]] = []

    for name, base_w, kws in tpl:
        w = base_w
        hit = False
        for kw in kws:
            if kw and kw in text:
                # small boost for each keyword hit (bounded)
                w += 0.08
                hit = True
                used.append(kw)
        boosted.append((name, w))

    # If no keyword hit at all, keep base weights but still return (for visualization)
    drivers = _normalize_weights(boosted)
    # de-duplicate keywords
    used2 = []
    seen = set()
    for k in used:
        if k not in seen:
            seen.add(k)
            used2.append(k)
    return drivers, used2


def _loss_components(label: str, total_p50: float) -> List[Tuple[str, float]]:
    weights = _COMPONENT_WEIGHTS_BY_LABEL.get(label) or [("减产损失", 0.75), ("品质/等级损失", 0.15), ("追加处置成本", 0.10)]
    ws = _normalize_weights([(n, w) for n, w in weights])
    comps = [(name, float(total_p50) * w) for name, w in ws]
    # ensure numeric stability
    return [(n, max(0.0, v)) for n, v in comps]


# -------------------------
# Forecast series
# -------------------------

def build_loss_forecast_series(
    res: EconomyResult,
    n_months: int = 6,
) -> Dict[str, Any]:
    final_p10 = float(res.total_loss_value_p10 or 0.0)
    final_p50 = float(res.total_loss_value_p50 or 0.0)
    final_p90 = float(res.total_loss_value_p90 or 0.0)

    weights = []
    taus = []
    for it in res.items:
        w = float(it.loss_value_p50 or 0.0)
        weights.append(w)
        taus.append(_hazard_tau_months(it.label))

    if sum(weights) > 1e-9:
        tau = sum(w * t for w, t in zip(weights, taus)) / sum(weights)
    else:
        tau = 2.0

    xs = list(range(n_months + 1))
    frac = [1.0 - math.exp(-x / max(tau, 1e-6)) for x in xs]

    base_p10 = [final_p10 * f for f in frac]
    base_p50 = [final_p50 * f for f in frac]
    base_p90 = [final_p90 * f for f in frac]

    if res.items and sum(weights) > 1e-9:
        red = sum((float(it.loss_value_p50 or 0.0) / sum(weights)) * _mitigation_reduction(it.label) for it in res.items)
    else:
        red = 0.2
    red = max(0.0, min(0.6, red))

    final_mit_p50 = final_p50 * (1.0 - red)
    final_mit_p10 = final_p10 * (1.0 - red)
    final_mit_p90 = final_p90 * (1.0 - red)

    mit_p50 = [final_mit_p50 * f for f in frac]
    mit_p10 = [final_mit_p10 * f for f in frac]
    mit_p90 = [final_mit_p90 * f for f in frac]

    return {
        "n_months": n_months,
        "tau_months": tau,
        "mitigation_reduction": red,
        "x_month_index": xs,
        "baseline": {"p10": base_p10, "p50": base_p50, "p90": base_p90},
        "mitigated": {"p10": mit_p10, "p50": mit_p50, "p90": mit_p90},
    }


# -------------------------
# Plotting
# -------------------------

def plot_loss_forecast(
    month_labels: List[str],
    series: Dict[str, Any],
    out_png: Path,
    title: str,
    y_label: str,
) -> None:
    _configure_chinese_fonts()
    _apply_readable_style()
    out_png.parent.mkdir(parents=True, exist_ok=True)

    x = list(range(len(month_labels)))
    base = series["baseline"]
    mit = series["mitigated"]

    plt.figure(figsize=(10, 5.2))
    plt.plot(x, base["p50"], marker="o", label="基线 P50")
    plt.fill_between(x, base["p10"], base["p90"], alpha=0.18, label="基线不确定性(P10-P90)")
    plt.plot(x, mit["p50"], marker="o", linestyle="--", label="处置后 P50(情景)")

    plt.xticks(x, month_labels, rotation=30, ha="right")
    plt.title(title)
    plt.ylabel(y_label)
    plt.xlabel("月份")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()


def plot_breakdown_bar(res: EconomyResult, out_png: Path, title: str) -> Dict[str, Any]:
    """
    Multi-label: horizontal bars sorted + uncertainty whiskers.
    Single-label: stacked components bar for richer interpretation.
    Returns detail dict for charts_data.json
    """
    _configure_chinese_fonts()
    _apply_readable_style()
    out_png.parent.mkdir(parents=True, exist_ok=True)

    detail: Dict[str, Any] = {"mode": "label_breakdown"}

    if not res.items:
        plt.figure(figsize=(9, 4.8))
        plt.title(title)
        plt.text(0.5, 0.5, "无可用损失分项数据", ha="center", va="center")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(out_png)
        plt.close()
        return detail

    # Single label -> stacked components (P50)
    if len(res.items) == 1:
        it = res.items[0]
        total = float(it.loss_value_p50 or 0.0)
        comps = _loss_components(it.label, total)
        comp_names = [c[0] for c in comps]
        comp_vals = [c[1] for c in comps]

        plt.figure(figsize=(10, 4.8))
        left = 0.0
        # stacked horizontal bar looks nicer for long labels
        for name, v in zip(comp_names, comp_vals):
            plt.barh([_label_display(it.label)], [v], left=left, label=name)
            left += v

        # annotate total
        plt.text(left * 1.01 if left > 0 else 0.01, 0, f"P50 合计：{left:,.2f}", va="center")

        plt.title(f"{title}｜细分为经济分项（P50，启发式）")
        plt.xlabel(_money_unit(res))
        plt.ylabel("灾害标签")
        plt.legend(ncol=3, frameon=False, loc="upper right")
        plt.tight_layout()
        plt.savefig(out_png)
        plt.close()

        detail["mode"] = "single_label_components"
        detail["label"] = it.label
        detail["components"] = [{"name": n, "value_p50": float(v)} for n, v in comps]
        return detail

    # Multi labels -> horizontal bars + whiskers (P10-P90)
    rows = []
    for it in res.items:
        p50 = float(it.loss_value_p50 or 0.0)
        p10 = float(it.loss_value_p10 or 0.0)
        p90 = float(it.loss_value_p90 or 0.0)
        rows.append((it.label, p50, p10, p90, float(it.confidence or 0.0)))

    rows.sort(key=lambda x: x[1], reverse=True)
    labels = [_label_display(r[0]) for r in rows]
    p50s = [r[1] for r in rows]
    p10s = [r[2] for r in rows]
    p90s = [r[3] for r in rows]

    # Asymmetric error bars: (p50-p10, p90-p50)
    xerr = [
        [max(0.0, p50 - p10) for p50, p10 in zip(p50s, p10s)],
        [max(0.0, p90 - p50) for p90, p50 in zip(p90s, p50s)],
    ]

    plt.figure(figsize=(10, 5.2))
    y = list(range(len(labels)))
    plt.barh(y, p50s)
    plt.errorbar(p50s, y, xerr=xerr, fmt="none", capsize=3)

    for yi, v in zip(y, p50s):
        plt.text(v * 1.01 if v > 0 else 0.01, yi, f"{v:,.2f}", va="center")

    plt.yticks(y, labels)
    plt.title(title + "（P50，含P10-P90不确定性）")
    plt.xlabel(_money_unit(res))
    plt.ylabel("灾害标签")
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()

    return detail


def plot_breakdown_pie(
    res: EconomyResult,
    out_png: Path,
    title: str,
    *,
    latest: Optional[Dict[str, Any]] = None,
    fallback_components_if_no_drivers: bool = True,
) -> Dict[str, Any]:
    """
    Multi-label: regular pie by labels.
    Single-label: donut by inferred drivers (heuristic; uses conclusion/notes keywords if available).
    Returns detail dict for charts_data.json.
    """
    _configure_chinese_fonts()
    _apply_readable_style()
    out_png.parent.mkdir(parents=True, exist_ok=True)

    detail: Dict[str, Any] = {"mode": "label_pie"}

    if not res.items:
        plt.figure(figsize=(8, 5.0))
        plt.title(title)
        plt.text(0.5, 0.5, "无可用数据", ha="center", va="center")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(out_png)
        plt.close()
        return detail

    # Multi-label: pie by labels
    if len(res.items) >= 2:
        labels = [_label_display(it.label) for it in res.items]
        vals = [max(0.0, float(it.loss_value_p50 or 0.0)) for it in res.items]
        s = sum(vals)
        if s <= 1e-9:
            vals = [1.0 for _ in labels]
        plt.figure(figsize=(8.2, 5.4))
        plt.pie(vals, labels=labels, autopct="%1.1f%%", startangle=90, pctdistance=0.8)
        plt.title(title)
        plt.tight_layout()
        plt.savefig(out_png)
        plt.close()
        return detail

    # Single-label: donut by drivers (preferred), else donut by components
    it = res.items[0]
    context_text = _extract_context_text(latest)
    drivers, used_keywords = _infer_drivers(it.label, context_text)

    if drivers:
        names = [d[0] for d in drivers]
        weights = [d[1] for d in drivers]
        plt.figure(figsize=(8.2, 5.4))
        wedges, texts, autotexts = plt.pie(
            weights,
            labels=names,
            autopct="%1.1f%%",
            startangle=90,
            pctdistance=0.78,
        )
        # donut hole
        centre_circle = plt.Circle((0, 0), 0.52, fc="white")
        plt.gca().add_artist(centre_circle)

        basis = "、".join(used_keywords[:6]) if used_keywords else "（无关键词命中，使用默认权重）"
        plt.title(f"{title}｜细分驱动因素（启发式）\n依据：{basis}")
        plt.tight_layout()
        plt.savefig(out_png)
        plt.close()

        detail["mode"] = "single_label_drivers"
        detail["label"] = it.label
        detail["basis_keywords"] = used_keywords
        detail["drivers"] = [{"name": n, "weight": float(w)} for n, w in drivers]
        return detail

    if fallback_components_if_no_drivers:
        total = float(it.loss_value_p50 or 0.0)
        comps = _loss_components(it.label, total)
        names = [c[0] for c in comps]
        vals = [c[1] for c in comps]
        s = sum(vals)
        if s <= 1e-9:
            vals = [1.0 for _ in names]

        plt.figure(figsize=(8.2, 5.4))
        plt.pie(vals, labels=names, autopct="%1.1f%%", startangle=90, pctdistance=0.78)
        centre_circle = plt.Circle((0, 0), 0.52, fc="white")
        plt.gca().add_artist(centre_circle)

        plt.title(f"{title}｜细分为经济分项（启发式）")
        plt.tight_layout()
        plt.savefig(out_png)
        plt.close()

        detail["mode"] = "single_label_components_pie"
        detail["label"] = it.label
        detail["components"] = [{"name": n, "value_p50": float(v)} for n, v in comps]
        return detail

    # last fallback: trivial pie
    plt.figure(figsize=(8.2, 5.4))
    plt.pie([1.0], labels=[_label_display(it.label)], autopct="%1.1f%%")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()
    return detail


# -------------------------
# JSON bundle
# -------------------------

def write_charts_data_json(
    out_json: Path,
    month_labels: List[str],
    series: Dict[str, Any],
    res: EconomyResult,
    *,
    single_detail: Optional[Dict[str, Any]] = None,
) -> None:
    out_json.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "schema": "agri_rag.economy_charts_data.v2",
        "time_utc": datetime.utcnow().isoformat() + "Z",
        "month_labels": month_labels,
        "loss_forecast": series,
        "breakdown": [
            {
                "label": it.label,
                "label_display": _label_display(it.label),
                "confidence": it.confidence,
                "affected_ratio": it.affected_ratio,
                "yield_loss_frac_p50": it.yield_loss_frac_p50,
                "loss_value_p50": it.loss_value_p50,
                "loss_value_p10": it.loss_value_p10,
                "loss_value_p90": it.loss_value_p90,
                "notes": it.notes,
            }
            for it in res.items
        ],
        "total": {
            "base_field_value_p50": res.base_field_value_p50,
            "total_loss_value_p10": res.total_loss_value_p10,
            "total_loss_value_p50": res.total_loss_value_p50,
            "total_loss_value_p90": res.total_loss_value_p90,
            "total_yield_loss_frac_p10": res.total_yield_loss_frac_p10,
            "total_yield_loss_frac_p50": res.total_yield_loss_frac_p50,
            "total_yield_loss_frac_p90": res.total_yield_loss_frac_p90,
            "unit": _money_unit(res),
        },
        "single_label_detail": single_detail or {},
    }
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


# -------------------------
# Public entry
# -------------------------

def generate_all_charts(
    res: EconomyResult,
    charts_dir: Path,
    n_months: int = 6,
    *,
    latest: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Generates charts and returns a dict describing chart paths and data.

    charts_dir is typically under output/assets/charts/<runid>_<timestamp>/
    but can be anywhere.

    NOTE:
    - latest is optional; only used to make SINGLE-LABEL charts richer (driver inference).
    - If you don't pass it, charts still work (fallback to component split).
    """
    charts_dir.mkdir(parents=True, exist_ok=True)

    month_labels = _month_labels(n_months=n_months)
    series = build_loss_forecast_series(res, n_months=n_months)

    loss_forecast_png = charts_dir / "loss_forecast.png"
    breakdown_bar_png = charts_dir / "loss_breakdown_bar.png"
    breakdown_pie_png = charts_dir / "loss_breakdown_pie.png"
    charts_data_json = charts_dir / "charts_data.json"

    plot_loss_forecast(
        month_labels=month_labels,
        series=series,
        out_png=loss_forecast_png,
        title="未来数月经济损失走向（累计口径，情景推演）",
        y_label=_money_unit(res),
    )

    single_detail_1 = plot_breakdown_bar(
        res=res,
        out_png=breakdown_bar_png,
        title="损失归因构成",
    )

    single_detail_2 = plot_breakdown_pie(
        res=res,
        out_png=breakdown_pie_png,
        title="损失构成占比",
        latest=latest,
        fallback_components_if_no_drivers=True,
    )

    single_detail: Dict[str, Any] = {}
    if len(res.items) == 1:
        # Merge two detail dicts for single-label explanations
        single_detail = {
            "bar_detail": single_detail_1,
            "pie_detail": single_detail_2,
        }

    write_charts_data_json(
        out_json=charts_data_json,
        month_labels=month_labels,
        series=series,
        res=res,
        single_detail=single_detail,
    )

    return {
        "charts_dir": str(charts_dir).replace("\\", "/"),
        "loss_forecast_png": str(loss_forecast_png).replace("\\", "/"),
        "breakdown_bar_png": str(breakdown_bar_png).replace("\\", "/"),
        "breakdown_pie_png": str(breakdown_pie_png).replace("\\", "/"),
        "charts_data_json": str(charts_data_json).replace("\\", "/"),
        "month_labels": month_labels,
        "series": series,
    }
