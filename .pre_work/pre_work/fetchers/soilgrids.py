from __future__ import annotations
import requests
from typing import Any, Dict, Optional
from .cache import SimpleCache

SOILGRIDS_URL = "https://rest.isric.org/soilgrids/v2.0/properties/query"
PROPS = ["clay","sand","silt","ocd","bdod"]

class SoilGridsClient:
    def __init__(self, cache: Optional[SimpleCache] = None, timeout: int = 30) -> None:
        self.cache = cache
        self.timeout = timeout

    def _get(self, params: Dict[str, Any]) -> Dict[str, Any]:
        key = SOILGRIDS_URL + "?" + "&".join([f"{k}={params[k]}" for k in sorted(params.keys())])
        if self.cache:
            hit = self.cache.get(key)
            if isinstance(hit, dict):
                return hit
        r = requests.get(SOILGRIDS_URL, params=params, timeout=self.timeout, headers={"User-Agent":"agri-pre-work-agent"})
        r.raise_for_status()
        data = r.json()
        if self.cache:
            self.cache.set(key, data)
        return data

    def fetch_profile(self, lat: float, lon: float, depth: str = "0-5cm") -> Dict[str, Any]:
        params = {"lat": lat, "lon": lon, "property": ",".join(PROPS), "depth": depth, "value": "mean"}
        return self._get(params)

    @staticmethod
    def parse_profile(payload: Dict[str, Any]) -> Dict[str, Optional[float]]:
        out: Dict[str, Optional[float]] = {"clay": None, "sand": None, "silt": None, "oc": None, "bdod": None}
        layers = ((payload or {}).get("properties") or {}).get("layers") or []
        for layer in layers:
            name = layer.get("name")
            depths = layer.get("depths") or []
            if not depths:
                continue
            vals = depths[0].get("values") or {}
            mean = vals.get("mean")
            if mean is None:
                continue
            if name == "clay":
                out["clay"] = float(mean)
            elif name == "sand":
                out["sand"] = float(mean)
            elif name == "silt":
                out["silt"] = float(mean)
            elif name == "ocd":
                out["oc"] = float(mean)
            elif name == "bdod":
                out["bdod"] = float(mean)
        return out
