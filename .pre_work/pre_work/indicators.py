from __future__ import annotations
from typing import Dict, List, Any
import numpy as np

def _rolling_sum(x: List[float], w: int) -> List[float]:
    out: List[float] = []
    s = 0.0
    q: List[float] = []
    for v in x:
        v = float(v)
        q.append(v); s += v
        if len(q) > w:
            s -= q.pop(0)
        out.append(s)
    return out

def _consec_days_below(x: List[float], thr: float) -> List[int]:
    out: List[int] = []
    c = 0
    for v in x:
        if float(v) < thr:
            c += 1
        else:
            c = 0
        out.append(c)
    return out

def _zscore_rolling(arr: List[float], w: int = 14) -> List[float]:
    out: List[float] = []
    for i in range(len(arr)):
        s = arr[max(0, i-w+1):i+1]
        mu = float(np.mean(s)) if s else 0.0
        sd = float(np.std(s)) if s else 1.0
        out.append((arr[i] - mu) / max(sd, 1e-6))
    return out

def compute_daily_indicators(daily: Dict[str, List[Any]]) -> Dict[str, List[float]]:
    p = [float(x or 0.0) for x in daily.get("precipitation_sum", [])]
    tmax = [float(x or 0.0) for x in daily.get("temperature_2m_max", [])]
    tmin = [float(x or 0.0) for x in daily.get("temperature_2m_min", [])]
    et0 = [float(x or 0.0) for x in daily.get("et0_fao_evapotranspiration", [])]
    vpd = [float(x or 0.0) for x in daily.get("vapor_pressure_deficit_max", [])]

    sm0 = [float(x or 0.0) for x in daily.get("soil_moisture_0_to_7cm_mean", [])]
    sm28 = [float(x or 0.0) for x in daily.get("soil_moisture_7_to_28cm_mean", [])]
    sm100 = [float(x or 0.0) for x in daily.get("soil_moisture_0_to_100cm_mean", [])]

    n = len(p)

    def pad(arr: List[float]) -> List[float]:
        if not arr:
            return [0.0]*n
        if len(arr) >= n:
            return arr[:n]
        last = arr[-1]
        return arr + [last]*(n-len(arr))

    sm0 = pad(sm0); sm28 = pad(sm28); sm100 = pad(sm100)

    wb = [p[i] - et0[i] for i in range(n)]
    wb7 = _rolling_sum(wb, 7)
    p3 = _rolling_sum(p, 3)
    p7 = _rolling_sum(p, 7)
    dry_days = _consec_days_below(p, thr=1.0)

    sm0_z = _zscore_rolling(sm0, 14) if n else []
    sm100_z = _zscore_rolling(sm100, 14) if n else []

    heat_degree = [max(0.0, tmax[i] - 35.0) for i in range(n)]
    frost_flag = [1.0 if tmin[i] <= 0.0 else 0.0 for i in range(n)]

    runoff_proxy = [max(0.0, p[i]) * (0.6 + 0.25*max(0.0, sm0_z[i])) for i in range(n)]

    return {
        "precip_mm": p,
        "precip_3d_mm": p3,
        "precip_7d_mm": p7,
        "et0_mm": et0,
        "water_balance_mm": wb,
        "water_balance_7d_mm": wb7,
        "vpd_kpa": vpd,
        "soil_moisture_0_7": sm0,
        "soil_moisture_7_28": sm28,
        "soil_moisture_0_100": sm100,
        "soil_moisture_0_7_z": sm0_z,
        "soil_moisture_0_100_z": sm100_z,
        "consec_dry_days": [float(x) for x in dry_days],
        "heat_degree": heat_degree,
        "frost_flag": frost_flag,
        "runoff_proxy": runoff_proxy,
        "tmax": tmax,
        "tmin": tmin,
    }
