import re
from typing import Any, Dict

_PATTERNS = [
    re.compile(r"\b\d+(\.\d+)?\s*(ml|mL|g|kg|L|l)\b"),
    re.compile(r"(兑水|配比|浓度|稀释|每亩|每公顷|喷施|喷雾)\s*[:：]?\s*\d"),
]

def redact_risky_details(text: str) -> str:
    if not text:
        return text
    t = text
    for p in _PATTERNS:
        t = p.sub("（已省略具体用量参数，建议咨询当地农技并按合规说明执行）", t)
    return t

def postprocess_advice_json(obj: Dict[str, Any]) -> Dict[str, Any]:
    def walk(x):
        if isinstance(x, str):
            return redact_risky_details(x)
        if isinstance(x, list):
            return [walk(v) for v in x]
        if isinstance(x, dict):
            return {k: walk(v) for k, v in x.items()}
        return x

    out = walk(obj)
    if isinstance(out, dict) and "disclaimer" not in out:
        out["disclaimer"] = "建议基于专家文档与模型输出生成，仅供辅助决策；涉及用药/工程改造/高风险操作请咨询当地农技与主管部门并遵循合规要求。"
    return out
