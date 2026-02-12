from __future__ import annotations
import json, time
from pathlib import Path
from typing import Any, Optional
from ..io_utils import ensure_dir, sha1

class SimpleCache:
    def __init__(self, cache_dir: str, ttl_hours: int = 12) -> None:
        self.dir = Path(cache_dir)
        self.ttl = max(1, ttl_hours) * 3600
        ensure_dir(self.dir)

    def _path(self, key: str) -> Path:
        return self.dir / f"{sha1(key)}.json"

    def get(self, key: str) -> Optional[Any]:
        p = self._path(key)
        if not p.exists():
            return None
        try:
            obj = json.loads(p.read_text(encoding="utf-8"))
            ts = float(obj.get("_ts", 0.0))
            if time.time() - ts > self.ttl:
                return None
            return obj.get("payload")
        except Exception:
            return None

    def set(self, key: str, payload: Any) -> None:
        p = self._path(key)
        data = {"_ts": time.time(), "payload": payload}
        p.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
