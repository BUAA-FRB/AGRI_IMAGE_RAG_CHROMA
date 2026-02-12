from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, List, Optional
from datetime import datetime, timedelta, timezone

from .config import SiteConfig, RunConfig
from .io_utils import write_json, write_text, now_ts, sha1
from .fetchers.cache import SimpleCache
from .fetchers.open_meteo import OpenMeteoClient
from .fetchers.soilgrids import SoilGridsClient
from .indicators import compute_daily_indicators
from .risk_models import score_risks
from .actions import suggest_actions_for_day
from .charts import generate_all_charts
from .prompt_builder import build_messages
from .report import build_report_md, write_next30_csv

def _parse_daily(payload: Dict[str, Any]) -> Dict[str, List[Any]]:
    daily = (payload or {}).get("daily") or {}
    if not isinstance(daily, dict):
        return {}
    return {k: list(v) for k, v in daily.items()}

def _extend_to_30_days_by_climatology(
    client: OpenMeteoClient,
    lat: float,
    lon: float,
    daily: Dict[str, List[Any]],
    target_total_days: int,
    model: str = "era5_land",
) -> Dict[str, Any]:
    dates = list(daily.get("time") or [])
    if not dates or len(dates) >= target_total_days:
        return {"daily": daily, "extended": False, "extended_days": 0}

    missing = target_total_days - len(dates)
    last_date = datetime.fromisoformat(dates[-1]).date()
    start_extra = last_date + timedelta(days=1)
    end_extra = last_date + timedelta(days=missing)

    vars_keep = ["precipitation_sum","temperature_2m_max","temperature_2m_min","et0_fao_evapotranspiration","vapor_pressure_deficit_max"]
    years = [start_extra.year - i for i in range(1, 6)]
    acc: Dict[str, List[float]] = {}
    ok = 0

    for y in years:
        s = start_extra.replace(year=y).isoformat()
        e = end_extra.replace(year=y).isoformat()
        try:
            arc = client.fetch_archive_range(lat, lon, s, e, timezone="UTC", model=model, daily_vars=["time"] + vars_keep)
            d = (arc or {}).get("daily") or {}
            if not d or "time" not in d:
                continue
            ok += 1
            for k in vars_keep:
                arr = d.get(k) or []
                if not arr:
                    continue
                arrf = [float(x or 0.0) for x in arr]
                if k not in acc:
                    acc[k] = arrf
                else:
                    m = min(len(acc[k]), len(arrf))
                    acc[k] = [acc[k][i] + arrf[i] for i in range(m)]
        except Exception:
            continue

    def _persist_last(daily: Dict[str, List[Any]], missing: int) -> None:
        for k in ["soil_moisture_0_to_7cm_mean","soil_moisture_7_to_28cm_mean","soil_moisture_28_to_100cm_mean","soil_moisture_0_to_100cm_mean","soil_temperature_0_to_7cm_mean","rain_sum","wind_gusts_10m_max"]:
            arr = daily.get(k) or []
            last = arr[-1] if arr else 0.0
            daily[k] = arr + [last]*missing

    if ok <= 0:
        for k in vars_keep:
            hist = [float(x or 0.0) for x in (daily.get(k) or [])]
            mu = sum(hist[-7:]) / max(1, len(hist[-7:]))
            daily[k] = (daily.get(k) or []) + [mu]*missing
        daily["time"] = dates + [(start_extra + timedelta(days=i)).isoformat() for i in range(missing)]
        _persist_last(daily, missing)
        return {"daily": daily, "extended": True, "extended_days": missing, "method": "persistence"}

    for k in vars_keep:
        avg = [v / ok for v in acc.get(k, [0.0]*missing)]
        daily[k] = (daily.get(k) or []) + avg[:missing]

    daily["time"] = dates + [(start_extra + timedelta(days=i)).isoformat() for i in range(missing)]
    _persist_last(daily, missing)
    return {"daily": daily, "extended": True, "extended_days": missing, "method": f"climatology_{ok}y"}

def _slice_tail_float(d: Dict[str, List[float]], tail: int) -> Dict[str, List[float]]:
    return {k: list(v[-tail:]) for k, v in d.items()}

def _build_next30_table(
    dates: List[str],
    ind: Dict[str, List[float]],
    risk: Dict[str, List[float]],
    levels: Dict[str, List[str]],
    triggers: Dict[str, List[str]],
) -> List[Dict[str, Any]]:
    n = len(dates)
    hazards = ["waterlogging","drought","heat","frost","disease_pressure","wind_lodging","operation_window"]
    rows: List[Dict[str, Any]] = []
    for i in range(n):
        r: Dict[str, Any] = {"date": dates[i]}
        for h in hazards:
            r[f"risk_{h}"] = f"{float(risk.get(h,[0.0]*n)[i]):.1f}"
            r[f"level_{h}"] = str(levels.get(h,["L0"]*n)[i])
            r[f"trig_{h}"] = str(triggers.get(h,[""]*n)[i])
        r["P(mm)"] = f"{float(ind.get('precip_mm',[0.0]*n)[i]):.1f}"
        r["ET0(mm)"] = f"{float(ind.get('et0_mm',[0.0]*n)[i]):.1f}"
        r["WB7(mm)"] = f"{float(ind.get('water_balance_7d_mm',[0.0]*n)[i]):.1f}"
        r["SM0-7(z)"] = f"{float(ind.get('soil_moisture_0_7_z',[0.0]*n)[i]):.2f}"
        rows.append(r)
    return rows

def run_site(site: SiteConfig, cfg: RunConfig, llm_runner: Optional[Any] = None) -> Path:
    run_id = sha1(f"{site.name}:{site.lat:.5f}:{site.lon:.5f}:{now_ts()}")[:10] + "_" + now_ts()
    out_dir = Path(cfg.out_root) / site.name / run_id
    charts_dir = out_dir / "assets" / "charts"
    out_dir.mkdir(parents=True, exist_ok=True)

    cache = SimpleCache(cfg.cache_dir, ttl_hours=cfg.cache_ttl_hours)
    om = OpenMeteoClient(cache=cache)

    payload = om.fetch_forecast_with_past(
        lat=site.lat, lon=site.lon,
        days_hist=cfg.days_hist,
        days_fore=cfg.days_fore,
        timezone=site.timezone,
        model=cfg.open_meteo_model,
    )
    daily = _parse_daily(payload)

    ext_meta = {"extended": False, "extended_days": 0}
    if cfg.extend_to_30:
        target_total = cfg.days_hist + 30
        ext = _extend_to_30_days_by_climatology(om, site.lat, site.lon, daily, target_total, model="era5_land")
        daily = ext.get("daily", daily)
        ext_meta = {k: ext.get(k) for k in ["extended","extended_days","method"] if k in ext}

    dates = list(daily.get("time") or [])

    soil: Dict[str, Any] = {}
    if cfg.use_soilgrids:
        try:
            sg = SoilGridsClient(cache=cache)
            s_payload = sg.fetch_profile(site.lat, site.lon, depth="0-5cm")
            soil = sg.parse_profile(s_payload)
            soil["source"] = "soilgrids"
        except Exception as e:
            soil = {"source": "soilgrids", "error": f"{type(e).__name__}: {e}"}

    ind = compute_daily_indicators(daily)

    thresholds = {
        "heavy_rain_mm_day": cfg.heavy_rain_mm_day,
        "dry_spell_days": cfg.dry_spell_days,
        "heat_c": cfg.heat_c,
        "frost_c": cfg.frost_c,
    }
    risk, levels, trig, summary = score_risks(ind, dates, soil=soil, cfg=thresholds)

    next30_dates = dates[-30:] if len(dates) >= 30 else dates
    tail = len(next30_dates)
    # slice tails
    ind30 = _slice_tail_float(ind, tail)
    risk30 = _slice_tail_float(risk, tail)
    levels30 = {k: list(v[-tail:]) for k, v in levels.items()}
    trig30 = {k: list(v[-tail:]) for k, v in trig.items()}
    next30_rows = _build_next30_table(next30_dates, ind30, risk30, levels30, trig30)

    actions_by_day: List[Dict[str, Any]] = []
    for i, d in enumerate(next30_dates):
        day_levels = {k: levels30.get(k, ["L0"]*tail)[i] for k in levels30.keys()}
        day_trig = {k: trig30.get(k, [""]*tail)[i] for k in trig30.keys()}
        acts = suggest_actions_for_day(day_levels, day_trig)
        actions_by_day.append({"date": d, "actions": acts})

    charts_meta = generate_all_charts(dates, ind, risk, levels, charts_dir)

    # LLM analysis
    llm_md = ""
    prompt_txt = ""
    if cfg.use_llm and llm_runner is not None:
        site_meta = {"name": site.name, "lat": site.lat, "lon": site.lon, "crop": site.crop, "stage": site.stage}
        bundle_for_llm = {"risk": {"summary": summary}, "tables": {"next30": next30_rows}, "thresholds": thresholds}
        messages = build_messages(site_meta, bundle_for_llm)
        prompt_txt = "\n\n".join([f"[{m['role']}]\n{m['content']}" for m in messages])
        llm_md = llm_runner.generate(messages)

    bundle = {
        "schema": "agri_pre_work.bundle.v1",
        "time_utc": datetime.now(timezone.utc).isoformat(),
        "site": {"name": site.name, "lat": site.lat, "lon": site.lon, "crop": site.crop, "stage": site.stage},
        "ext": ext_meta,
        "thresholds": thresholds,
        "soil": soil,
        "weather": {"meta": {"timezone": payload.get("timezone")}, "daily": daily},
        "indicators": ind,
        "risk": {"daily_risk": risk, "daily_level": levels, "triggers": trig, "summary": summary},
        "tables": {"next30": next30_rows},
        "actions": {"by_day": actions_by_day},
        "charts": charts_meta,
    }
    write_json(out_dir / "bundle.json", bundle)
    write_next30_csv(out_dir / "risk_table_next30.csv", next30_rows)

    if prompt_txt:
        write_text(out_dir / "risk_prompt.txt", prompt_txt)
    if llm_md:
        write_text(out_dir / "llm_analysis.md", llm_md)

    report_md = build_report_md(bundle["site"], bundle, out_dir, llm_md=llm_md if llm_md else None)
    write_text(out_dir / "risk_report.md", report_md)
    return out_dir
