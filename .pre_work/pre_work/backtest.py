from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime, timezone
import math
import json

from .fetchers.cache import SimpleCache
from .fetchers.open_meteo import OpenMeteoClient
from .indicators import compute_daily_indicators
from .risk_models import score_risks
from .io_utils import write_json, write_text

def _level_to_int(x: str) -> int:
    return {"L0":0,"L1":1,"L2":2,"L3":3}.get(x, 0)

def _brier(probs: List[float], y: List[int]) -> float:
    if not probs or not y:
        return float("nan")
    n = min(len(probs), len(y))
    s = 0.0
    for i in range(n):
        p = max(0.0, min(1.0, probs[i]))
        s += (p - float(y[i]))**2
    return s / n

def backtest_from_bundle(
    bundle_path: Path,
    out_dir: Path,
    cache_dir: str = ".pre_work/.cache",
    cache_ttl_hours: int = 12,
    archive_model: str = "era5_land",
) -> Path:
    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    site = bundle.get("site") or {}
    thresholds = bundle.get("thresholds") or {}
    lat = float(site.get("lat")); lon = float(site.get("lon"))
    pred_next30 = (bundle.get("tables") or {}).get("next30") or []
    if not pred_next30:
        raise ValueError("bundle.tables.next30 is empty, cannot backtest")

    dates = [r["date"] for r in pred_next30]
    start = dates[0]
    end = dates[-1]

    cache = SimpleCache(cache_dir, ttl_hours=cache_ttl_hours)
    om = OpenMeteoClient(cache=cache)

    # Fetch realized weather in the target window using archive API (reanalysis)
    arc = om.fetch_archive_range(lat, lon, start, end, timezone="UTC", model=archive_model)
    daily = (arc or {}).get("daily") or {}
    daily = {k: list(v) for k, v in daily.items()} if isinstance(daily, dict) else {}
    real_dates = list(daily.get("time") or [])
    if not real_dates:
        raise RuntimeError("archive daily.time empty; API may reject the range")

    ind = compute_daily_indicators(daily)
    risk, levels, trig, summary = score_risks(ind, real_dates, soil=bundle.get("soil") or {}, cfg=thresholds)

    # Align (by date string)
    idx = {d:i for i,d in enumerate(real_dates)}
    hazards = ["waterlogging","drought","heat","frost","disease_pressure","wind_lodging"]
    metrics: Dict[str, Any] = {"schema":"agri_pre_work.backtest.v1", "time_utc": datetime.now(timezone.utc).isoformat()}
    per_h: Dict[str, Any] = {}

    rows_cmp: List[Dict[str, Any]] = []
    for d in dates:
        if d not in idx:
            continue
        i = idx[d]
        r: Dict[str, Any] = {"date": d}
        for h in hazards:
            # predicted level from table
            pred_level = next((row.get(f"level_{h}") for row in pred_next30 if row.get("date")==d), "L0")
            real_level = levels.get(h, ["L0"]*len(real_dates))[i]
            pred_risk = float(next((row.get(f"risk_{h}") for row in pred_next30 if row.get("date")==d), 0.0))
            real_risk = float(risk.get(h, [0.0]*len(real_dates))[i])
            r[f"pred_{h}_risk"] = pred_risk
            r[f"real_{h}_risk"] = real_risk
            r[f"pred_{h}_level"] = pred_level
            r[f"real_{h}_level"] = real_level
        rows_cmp.append(r)

    # Compute simple event metrics: treat L2+ as event
    for h in hazards:
        pred_prob = []
        real_y = []
        abs_err = []
        for row in rows_cmp:
            pred = float(row.get(f"pred_{h}_risk", 0.0))/100.0
            real_evt = 1 if _level_to_int(str(row.get(f"real_{h}_level","L0"))) >= 2 else 0
            pred_evt = 1 if _level_to_int(str(row.get(f"pred_{h}_level","L0"))) >= 2 else 0
            pred_prob.append(pred)
            real_y.append(real_evt)
            abs_err.append(abs(float(row.get(f"pred_{h}_risk",0.0)) - float(row.get(f"real_{h}_risk",0.0))))
        # precision/recall
        tp = sum(1 for i in range(len(real_y)) if real_y[i]==1 and pred_prob[i]>=0.6)
        fp = sum(1 for i in range(len(real_y)) if real_y[i]==0 and pred_prob[i]>=0.6)
        fn = sum(1 for i in range(len(real_y)) if real_y[i]==1 and pred_prob[i]<0.6)
        precision = tp / (tp+fp) if (tp+fp)>0 else float("nan")
        recall = tp / (tp+fn) if (tp+fn)>0 else float("nan")
        per_h[h] = {
            "mean_abs_error": sum(abs_err)/max(1,len(abs_err)),
            "brier_L2plus": _brier(pred_prob, real_y),
            "precision_at_0.6": precision,
            "recall_at_0.6": recall,
            "real_peak": float(max(risk.get(h,[0.0]))),
            "real_peak_day": real_dates[risk.get(h,[0.0]).index(max(risk.get(h,[0.0])))] if risk.get(h) else None,
        }

    metrics["per_hazard"] = per_h
    metrics["range"] = {"start": start, "end": end, "archive_model": archive_model}
    metrics["site"] = site

    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "backtest_metrics.json", metrics)
    write_json(out_dir / "backtest_compare_rows.json", rows_cmp)

    # markdown report
    md: List[str] = []
    md.append("# 预警回测报告（灾后）")
    md.append(f"> 回测区间：{start} ~ {end}（archive={archive_model}）")
    md.append("")
    md.append("## 1. 结论概览")
    md.append("- 本回测使用再分析/历史数据重算风险曲线，与预警阶段输出对齐对比。")
    md.append("- 指标含义：MAE=风险分绝对误差均值；Brier(L2+)=把风险分当事件概率的校准误差；Precision/Recall 在阈值0.6处计算。")
    md.append("")
    md.append("## 2. 各类风险回测指标")
    md.append("| hazard | MAE | Brier(L2+) | Precision@0.6 | Recall@0.6 | RealPeak | RealPeakDay |")
    md.append("|---|---:|---:|---:|---:|---:|---|")
    for h, m in per_h.items():
        md.append(f"| {h} | {m['mean_abs_error']:.2f} | {m['brier_L2plus']:.3f} | {m['precision_at_0.6'] if not math.isnan(m['precision_at_0.6']) else 'NA'} | {m['recall_at_0.6'] if not math.isnan(m['recall_at_0.6']) else 'NA'} | {m['real_peak']:.1f} | {m['real_peak_day']} |")
    md.append("")
    md.append("## 3. 建议的校准方向（下一轮预警更准）")
    md.append("- 若涝风险高估：优先引入地块排水条件、地形低洼掩膜、沟渠/泵站能力做校准因子。")
    md.append("- 若旱风险低估：补充灌溉记录与真实土壤质地（砂/黏比例）与根区含水量传感器。")
    md.append("- 若热/霜误报：用作物具体生育期（敏感期）与微气候（林带/棚膜）修正阈值。")
    md.append("")
    write_text(out_dir / "backtest_report.md", "\n".join(md) + "\n")
    return out_dir
