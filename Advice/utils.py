import os
import json
import datetime
from typing import Any, Dict


def ensure_dir(p: str) -> None:
    if p:
        os.makedirs(p, exist_ok=True)


def read_json(path: str) -> Dict[str, Any]:
    ap = os.path.abspath(path)
    if not os.path.exists(ap):
        raise FileNotFoundError(f"read_json: file not found: {ap}")
    with open(ap, "r", encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, dict):
        raise ValueError(f"read_json: expected dict json, got {type(obj)} from {ap}")
    return obj


def write_json(path: str, obj: Any, indent: int = 2) -> None:
    """
    default=str：避免 numpy.float32 / pathlib / datetime 等不可序列化对象
    导致写文件失败（这类失败经常让你看到“只建目录没文件”）。
    """
    ensure_dir(os.path.dirname(path) or ".")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=indent, default=str)


def now_utc_iso() -> str:
    return datetime.datetime.utcnow().replace(tzinfo=datetime.timezone.utc).isoformat().replace("+00:00", "Z")
