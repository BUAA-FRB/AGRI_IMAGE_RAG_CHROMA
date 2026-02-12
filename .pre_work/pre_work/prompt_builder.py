from __future__ import annotations
import json
from typing import Any, Dict, List

def _compact_json(obj: Any, max_chars: int = 12000) -> str:
    s = json.dumps(obj, ensure_ascii=False, indent=2)
    if len(s) <= max_chars:
        return s
    return s[:max_chars] + "\n...<truncated>...\n"

_SYSTEM = (
    "你是一名‘农业气象-水文-土壤风险预警’专家，写作对象是地块管理与现场处置人员。\n"
    "你只使用输入的结构化数据进行解释，不编造数字；不确定就说明不确定，并给出需要补充的数据。\n\n"
    "【输出格式硬性规范（必须遵守）】\n"
    "1) 只输出纯 Markdown 正文：禁止输出任何代码围栏/代码块（包括 ``` 等）。\n"
    "2) 不要输出 H1 标题（# ...），外层模板会提供。\n"
    "3) 列表格式：有序仅用 1. 2. 3.；无序仅用 - 。\n"
    "4) 不要粘贴输入 JSON；只写解释 + 行动窗口 + 核验要点。\n"
)

def build_messages(site_meta: Dict[str, Any], bundle: Dict[str, Any]) -> List[Dict[str, str]]:
    summary = bundle.get("risk", {}).get("summary") or {}
    next30 = bundle.get("tables", {}).get("next30") or []
    next30_small = next30[:12]
    thresholds = bundle.get("thresholds", {})

    user = f"""请基于以下结构化信息，撰写报告的第 3 节《风险解读与行动窗口（大模型解读）》。

【你要输出的章节结构（严格按此输出）】
### 3.1 未来 7–30 天总体判断（1 段）
### 3.2 风险时间表与行动窗口（要点）
### 3.3 需要现场核验的关键点（要点）
### 3.4 不确定性与数据补充清单（要点）

—— 输入数据（仅用于分析，禁止原样复述）——
A) 地块信息
{_compact_json(site_meta, 4000)}

B) 风险摘要
{_compact_json(summary, 7000)}

C) 未来 30 天节选表（前 12 行）
{_compact_json(next30_small, 7000)}

D) 关键阈值
{_compact_json(thresholds, 2000)}
"""
    return [{"role":"system","content":_SYSTEM},{"role":"user","content":user}]
