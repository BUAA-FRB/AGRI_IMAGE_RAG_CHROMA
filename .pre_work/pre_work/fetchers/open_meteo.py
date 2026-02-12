from __future__ import annotations
import requests
from typing import Any, Dict, List, Optional
from .cache import SimpleCache

FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
ARCHIVE_URL  = "https://archive-api.open-meteo.com/v1/archive"

DEFAULT_DAILY = [
    "precipitation_sum","rain_sum",
    "temperature_2m_max","temperature_2m_min",
    "wind_gusts_10m_max",
    "et0_fao_evapotranspiration",
    "vapor_pressure_deficit_max",
    "soil_moisture_0_to_7cm_mean","soil_moisture_7_to_28cm_mean",
    "soil_moisture_28_to_100cm_mean","soil_moisture_0_to_100cm_mean",
    "soil_temperature_0_to_7cm_mean",
]

DEFAULT_HOURLY = [
    "temperature_2m","relative_humidity_2m","precipitation","wind_gusts_10m","soil_moisture_0_to_7cm",
]

class OpenMeteoClient:
    def __init__(self, cache: Optional[SimpleCache] = None, timeout: int = 30) -> None:
        self.cache = cache
        self.timeout = timeout

    def _get(self, url: str, params: Dict[str, Any]) -> Dict[str, Any]:
        key = url + "?" + "&".join([f"{k}={params[k]}" for k in sorted(params.keys())])
        if self.cache:
            hit = self.cache.get(key)
            if isinstance(hit, dict):
                return hit
        r = requests.get(url, params=params, timeout=self.timeout)
        r.raise_for_status()
        data = r.json()
        if self.cache:
            self.cache.set(key, data)
        return data

    def fetch_forecast_with_past(
        self,
        lat: float, lon: float,
        days_hist: int, days_fore: int,
        timezone: str = "auto",
        model: str = "best_match",
        daily_vars: Optional[List[str]] = None,
        hourly_vars: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        daily_vars = daily_vars or DEFAULT_DAILY
        hourly_vars = hourly_vars or DEFAULT_HOURLY
        params = {
            "latitude": lat, "longitude": lon,
            "timezone": timezone,
            "models": model,
            "daily": ",".join(daily_vars),
            "hourly": ",".join(hourly_vars),
            "forecast_days": int(days_fore),
            "past_days": int(days_hist),
        }
        return self._get(FORECAST_URL, params)

    def fetch_archive_range(
        self,
        lat: float, lon: float,
        start_date: str, end_date: str,
        timezone: str = "UTC",
        model: str = "era5_land",
        daily_vars: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        daily_vars = daily_vars or DEFAULT_DAILY
        params = {
            "latitude": lat, "longitude": lon,
            "timezone": timezone,
            "models": model,
            "start_date": start_date,
            "end_date": end_date,
            "daily": ",".join(daily_vars),
        }
        return self._get(ARCHIVE_URL, params)
