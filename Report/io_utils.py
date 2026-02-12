from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Union


def read_json(path: Union[str, Path]) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"JSON not found: {p}")
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_text(path: Union[str, Path], text: str) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")


def safe_str(x: Any) -> str:
    if x is None:
        return ""
    return str(x)


def to_float(x: Any):
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None
