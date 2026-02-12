import re
from typing import Dict, List


_DISALLOWED_PATTERNS = [
    r"\b\d+(\.\d+)?\s*(mg|g|kg|ml|mL|L|ppm|%)\b",
    r"\b\d+(\.\d+)?\s*(mg/L|g/L|ml/L|mL/L|kg/ha|L/ha)\b",
    r"(配比|兑水|稀释|浓度|用量|剂量|喷施|施药|农药|杀虫剂|杀菌剂|除草剂)",
]


def sanitize_text(s: str) -> str:
    for pat in _DISALLOWED_PATTERNS:
        if re.search(pat, s, flags=re.IGNORECASE):
            return "涉及任何药剂/化学品/器械与高风险操作：请由具备资质人员按产品标签与当地法规执行，并落实防护。"
    return s


def sanitize_actions(actions: List[str]) -> List[str]:
    return [sanitize_text(a) for a in actions]


def add_global_disclaimer(strategy: Dict) -> Dict:
    strategy.setdefault("disclaimer", [])
    if not strategy["disclaimer"]:
        strategy["disclaimer"] = [
            "本策略为流程化建议，用于辅助巡查、记录与处置编排，不替代当地农技/植保/水利等专业指导。",
            "涉及任何药剂、化学品、机械设备与高风险操作时，应由具备资质人员按标签说明与当地法规执行，并做好个人防护。",
            "模型预测与证据检索存在不确定性；建议结合多时相/多光谱/现场巡查进行复核。"
        ]
    return strategy
