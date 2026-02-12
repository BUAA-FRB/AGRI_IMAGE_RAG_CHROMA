import json
from typing import Any, Dict, Optional

from .qwen25_client import Qwen25Client, Qwen25GenConfig


_PROMPT = """你是农业灾害智能体的“策略整理助手”。输入是一个JSON策略（灾前/灾中/灾后）。
任务：
1) 给出一句话“关键结论”（不夸大、不臆测）
2) 对每个阶段给出 Top-3 优先事项（简短要点）
3) 给出“需要补采/补充的关键信息清单”
严格约束：
- 不要给出任何药剂/化学品/器械的具体用量、配比、操作步骤细节
- 如需涉及高风险操作，只写“由具备资质人员按标签与法规执行”
输出必须是严格JSON，格式如下：
{
  "key_takeaway": "...",
  "priorities": {"灾前":[...], "灾中":[...], "灾后":[...]},
  "data_requests": [...]
}
下面是输入JSON：
"""


def _extract_first_json(text: str) -> Dict[str, Any]:
    # 简单容错：从第一个 { 到最后一个 } 截取
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        cand = text[start:end+1]
        try:
            return json.loads(cand)
        except Exception:
            pass
    return {}


def refine_strategy_with_qwen25(
    strategy: Dict[str, Any],
    client: Qwen25Client,
    gen: Optional[Qwen25GenConfig] = None,
) -> Dict[str, Any]:
    prompt = _PROMPT + json.dumps(strategy, ensure_ascii=False, indent=2)
    raw = client.generate(prompt, gen=gen)
    obj = _extract_first_json(raw)
    obj.setdefault("key_takeaway", "")
    obj.setdefault("priorities", {"灾前": [], "灾中": [], "灾后": []})
    obj.setdefault("data_requests", [])
    return {"raw_text": raw, "json": obj}
