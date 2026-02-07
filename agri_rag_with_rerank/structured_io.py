from __future__ import annotations

import json
import re
from typing import Any, Dict, Optional

_CODE_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)


def _strip_code_fence(text: str) -> str:
    m = _CODE_FENCE_RE.search(text)
    if m:
        return m.group(1).strip()
    return text.strip()


def _remove_trailing_commas(s: str) -> str:
    s = re.sub(r",\s*}", "}", s)
    s = re.sub(r",\s*]", "]", s)
    return s


def _strip_common_prefixes(s: str) -> str:
    # 常见的“JSON:” “Output:” 等前缀
    t = s.strip()
    for pfx in ["JSON:", "json:", "Output:", "output:", "结果:", "输出:"]:
        if t.startswith(pfx):
            return t[len(pfx):].strip()
    return t


def _try_json_loads_dict_or_inner_dict(s: str) -> Optional[Dict[str, Any]]:
    try:
        obj = json.loads(s)
    except Exception:
        return None

    if isinstance(obj, dict):
        return obj

    # Sometimes model outputs a JSON string: "{...}" or "{\"a\":1}"
    if isinstance(obj, str):
        inner = obj.strip()
        if inner.startswith("{") and inner.endswith("}"):
            try:
                inner2 = _remove_trailing_commas(inner)
                obj2 = json.loads(inner2)
                if isinstance(obj2, dict):
                    return obj2
            except Exception:
                return None

    return None


def _scan_first_balanced_json_object(t: str) -> Optional[Dict[str, Any]]:
    start = -1
    depth = 0
    in_str = False
    esc = False

    for i, ch in enumerate(t):
        if in_str:
            if esc:
                esc = False
                continue
            if ch == "\\":
                esc = True
                continue
            if ch == '"':
                in_str = False
            continue

        if ch == '"':
            in_str = True
            continue

        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
            continue

        if ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start != -1:
                    cand = t[start : i + 1].strip()
                    cand = _remove_trailing_commas(cand)
                    obj = _try_json_loads_dict_or_inner_dict(cand)
                    if obj is not None:
                        return obj
                    start = -1
            continue

    return None


def extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None

    # remove BOM / invisible
    t = text.replace("\ufeff", "").replace("\u200b", "")
    t = _strip_code_fence(t).strip()
    t = _strip_common_prefixes(t)
    t = t.strip()

    # 1) direct load (dict or json-string-that-contains-dict)
    obj = _try_json_loads_dict_or_inner_dict(_remove_trailing_commas(t))
    if obj is not None:
        return obj

    # 2) greedy: take substring from first { to last }
    l = t.find("{")
    r = t.rfind("}")
    if l != -1 and r != -1 and r > l:
        cand = t[l : r + 1].strip()
        cand = _remove_trailing_commas(cand)
        obj = _try_json_loads_dict_or_inner_dict(cand)
        if obj is not None:
            return obj

    # 3) balanced scan: extract the first brace-balanced object
    obj = _scan_first_balanced_json_object(t)
    if obj is not None:
        return obj

    return None


def normalize_predicted_labels(obj: Dict[str, Any]) -> list[str]:
    out: list[str] = []

    if not isinstance(obj, dict):
        return out

    pls = obj.get("predicted_labels", None)
    if isinstance(pls, list):
        for it in pls:
            if isinstance(it, dict):
                lb = it.get("label", None)
                if isinstance(lb, str) and lb.strip():
                    out.append(lb.strip())
            elif isinstance(it, str) and it.strip():
                out.append(it.strip())

    if out:
        return dedup_keep_order(out)

    conc = obj.get("conclusion", None)
    if isinstance(conc, str) and conc.strip():
        parts = re.split(r"[;,，、|/\n]+", conc)
        parts = [p.strip() for p in parts if p.strip()]
        return dedup_keep_order(parts)

    return out


def dedup_keep_order(xs: list[str]) -> list[str]:
    seen = set()
    out: list[str] = []
    for x in xs:
        if x not in seen:
            out.append(x)
            seen.add(x)
    return out
