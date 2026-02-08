import json
from typing import Any, Dict, Optional

def extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    s = text.strip()

    if "```" in s:
        parts = s.split("```")
        for p in parts:
            if "{" in p and "}" in p:
                s = p.strip()
                break

    start = s.find("{")
    if start < 0:
        return None

    depth = 0
    end = -1
    for i in range(start, len(s)):
        if s[i] == "{":
            depth += 1
        elif s[i] == "}":
            depth -= 1
            if depth == 0:
                end = i + 1
                break
    if end < 0:
        return None

    cand = s[start:end].strip()
    try:
        return json.loads(cand)
    except Exception:
        cand2 = cand.replace(",}", "}").replace(",]", "]")
        try:
            return json.loads(cand2)
        except Exception:
            return None
