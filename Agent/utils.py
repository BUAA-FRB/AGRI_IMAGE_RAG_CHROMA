import json
import os
import time
import uuid
from typing import Any, Dict, Optional


def utc_ts() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def new_run_id(prefix: str = "agent") -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def ensure_dir(path: str) -> str:
    if path:
        os.makedirs(path, exist_ok=True)
    return path


def read_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: str, obj: Dict[str, Any]) -> None:
    ensure_dir(os.path.dirname(path) or ".")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def try_float(x: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        return float(x)
    except Exception:
        return default


def safe_relpath(target: str, base_dir: str) -> str:
    try:
        return os.path.relpath(target, start=base_dir)
    except Exception:
        return target
