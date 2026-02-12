from __future__ import annotations
from typing import Dict, List, Any, Tuple, Optional
import math

def _clip(x: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return max(lo, min(hi, x))

def _level(score: float) -> str:
    if score >= 80: return "L3"
    if score >= 60: return "L2"
    if score >= 35: return "L1"
    return "L0"

def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))

def score_risks(
    ind: Dict[str, List[float]],
    dates: List[str],
    soil: Optional[Dict[str, Any]] = None,
    cfg: Optional[Dict[str, Any]] = None,
) -> Tuple[Dict[str, List[float]], Dict[str, List[str]], Dict[str, List[str]], Dict[str, Any]]:
    cfg = cfg or {}
    heavy_rain = float(cfg.get("heavy_rain_mm_day", 50.0))
    dry_days_thr = int(cfg.get("dry_spell_days", 7))
    frost_c = float(cfg.get("frost_c", 0.0))
    heat_c = float(cfg.get("heat_c", 35.0))

    clay = float((soil or {}).get("clay") or 0.0)
    sand = float((soil or {}).get("sand") or 0.0)
    clay_factor = _clip((clay - 25.0) * 1.2 + 50.0) / 100.0
    sand_factor = _clip((sand - 40.0) * 1.2 + 50.0) / 100.0

    n = len(dates)
    p = ind.get("precip_mm", [0.0]*n)
    p3 = ind.get("precip_3d_mm", [0.0]*n)
    wb7 = ind.get("water_balance_7d_mm", [0.0]*n)
    sm0z = ind.get("soil_moisture_0_7_z", [0.0]*n)
    sm100z = ind.get("soil_moisture_0_100_z", [0.0]*n)
    dry = ind.get("consec_dry_days", [0.0]*n)
    tmax = ind.get("tmax", [0.0]*n)
    tmin = ind.get("tmin", [0.0]*n)
    vpd = ind.get("vpd_kpa", [0.0]*n)
    runoff = ind.get("runoff_proxy", [0.0]*n)
    gust = [float(x or 0.0) for x in ind.get("wind_gusts_10m_max", [0.0]*n)] if "wind_gusts_10m_max" in ind else [0.0]*n

    keys = ["waterlogging","drought","heat","frost","wind_lodging","disease_pressure","operation_window"]
    daily_risk: Dict[str, List[float]] = {k: [] for k in keys}
    daily_level: Dict[str, List[str]] = {k: [] for k in keys}
    triggers: Dict[str, List[str]] = {k: [] for k in keys}

    for i in range(n):
        wet_driver = (
            0.40 * _sigmoid((p[i] - heavy_rain)/10.0) +
            0.20 * _sigmoid((p3[i] - heavy_rain)/15.0) +
            0.20 * _sigmoid(sm0z[i]) +
            0.20 * _sigmoid((runoff[i] - 20.0)/10.0)
        )
        waterlogging = 100.0 * wet_driver * (0.7 + 0.6*clay_factor)

        dry_driver = (
            0.40 * _sigmoid((-wb7[i] - 10.0)/10.0) +
            0.30 * _sigmoid((dry[i] - dry_days_thr)/2.0) +
            0.30 * _sigmoid(-sm100z[i])
        )
        drought = 100.0 * dry_driver * (0.75 + 0.5*sand_factor)

        heat_driver = 0.65 * _sigmoid((tmax[i] - heat_c)/2.5) + 0.35 * _sigmoid((vpd[i] - 1.6)/0.4)
        heat = 100.0 * heat_driver
        frost = 100.0 * _sigmoid((frost_c - tmin[i])/1.5)
        wind = 100.0 * _sigmoid((gust[i] - 60.0)/10.0)
        disease = 100.0 * (0.6*_sigmoid((25.0 - abs((tmax[i]+tmin[i])/2.0 - 22.0))/3.0) + 0.4*_sigmoid((1.2 - vpd[i])/0.3))
        op = 100.0 * (1.0 - 0.6*_sigmoid((p[i]-10.0)/5.0) - 0.2*_sigmoid(abs(((tmax[i]+tmin[i])/2.0)-20.0)/5.0) - 0.2*_sigmoid(abs(sm0z[i])/1.0))
        op = _clip(op)

        scores = {
            "waterlogging": _clip(waterlogging),
            "drought": _clip(drought),
            "heat": _clip(heat),
            "frost": _clip(frost),
            "wind_lodging": _clip(wind),
            "disease_pressure": _clip(disease),
            "operation_window": _clip(op),
        }

        trig_map = {k: [] for k in keys}
        if p[i] >= heavy_rain: trig_map["waterlogging"].append(f"日降雨≥{heavy_rain:.0f}mm")
        if p3[i] >= heavy_rain: trig_map["waterlogging"].append("3日累计偏大")
        if sm0z[i] >= 1.0: trig_map["waterlogging"].append("表层土壤偏湿(>+1σ)")
        if dry[i] >= dry_days_thr: trig_map["drought"].append(f"连旱≥{dry_days_thr}天")
        if wb7[i] <= -15.0: trig_map["drought"].append("7日水分收支偏负")
        if tmax[i] >= heat_c: trig_map["heat"].append(f"最高温≥{heat_c:.0f}℃")
        if vpd[i] >= 1.8: trig_map["heat"].append("VPD偏高(蒸散压强)")
        if tmin[i] <= frost_c: trig_map["frost"].append(f"最低温≤{frost_c:.0f}℃")
        if gust[i] >= 70.0: trig_map["wind_lodging"].append("阵风偏强(≥70)")
        if vpd[i] <= 1.0 and p[i] >= 2.0: trig_map["disease_pressure"].append("湿热+降水(病害条件)")
        if p[i] <= 2.0 and 12.0 <= (tmax[i]+tmin[i])/2.0 <= 26.0: trig_map["operation_window"].append("适合作业窗口")

        for k in keys:
            daily_risk[k].append(scores[k])
            daily_level[k].append(_level(scores[k]))
            triggers[k].append(";".join(trig_map[k]) if trig_map[k] else "")

    summary: Dict[str, Any] = {}
    for k, arr in daily_risk.items():
        if not arr:
            continue
        peak = max(arr)
        peak_day = dates[arr.index(peak)]
        first_L2 = next((dates[i] for i,s in enumerate(arr) if s>=60), None)
        first_L3 = next((dates[i] for i,s in enumerate(arr) if s>=80), None)
        summary[k] = {"peak": peak, "peak_day": peak_day, "first_L2": first_L2, "first_L3": first_L3}

    return daily_risk, daily_level, triggers, summary
