from typing import Any, Dict, List

def build_messages(pred: Dict[str, Any], evidence_hint: Dict[str, Any], retrieved: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    kb_lines = []
    for i, r in enumerate(retrieved, start=1):
        meta = r.get("meta") or {}
        title = meta.get("title") or ""
        section = meta.get("section") or ""
        src = meta.get("source") or meta.get("path") or ""
        kb_lines.append(
            f"[K{i}] kid={r.get('kid','')} | title={title} | section={section} | source={src}\n{(r.get('text') or '').strip()}"
        )
    kb_block = "\n\n".join(kb_lines) if kb_lines else "（未检索到专家文档片段）"

    system = (
        "你是一名农业灾害应急与田间管理专家。你将基于“模型预测结果”和“检索到的专家文档片段”生成专业建议。\n"
        "强制约束：\n"
        "1) 不输出任何化学药剂的具体剂量/配比/喷施参数等细节；如涉及用药，仅给出原则性建议并提示咨询当地农技与合规说明。\n"
        "2) 输出必须是严格 JSON（不要附加解释性文字、不要 markdown）。\n"
        "3) 建议分为：灾前/灾中/灾后，并包含：损失统计方案、灾后重建要点。\n"
        "4) 关键建议尽量引用 [K1][K2]…（没有就不硬编）。"
    )

    user = f"""
【模型预测摘要】
label: {pred.get("pred_label","")}
confidence: {pred.get("confidence",0):.2f}
conclusion: {pred.get("conclusion","")}
uncertainty: {pred.get("uncertainty","")}

【证据提示（可选）】
{evidence_hint}

【专家文档片段（可引用）】
{kb_block}

【请输出严格 JSON，schema 如下】
{{
  "schema": "advice.expert_plan.v1",
  "summary": {{
    "label": "...",
    "confidence": 0.xx,
    "risk_level": "low|medium|high|needs_review",
    "key_takeaway": "一句话总结"
  }},
  "phases": {{
    "灾前": [{{"action":"...","why":"...","refs":["K1"]}}],
    "灾中": [{{"action":"...","why":"...","refs":["K2"]}}],
    "灾后": [{{"action":"...","why":"...","refs":["K3"]}}]
  }},
  "loss_assessment": {{
    "what_to_measure": ["..."],
    "how_to_record": ["..."],
    "outputs": ["表格/口径/照片取证等"],
    "refs": ["K1"]
  }},
  "recovery_rebuild": {{
    "priority_steps": ["..."],
    "monitoring_plan": ["..."],
    "refs": ["K2"]
  }},
  "data_requests": ["还缺哪些信息可以显著提高建议质量"],
  "disclaimer": "安全与合规提示"
}}
""".strip()

    return [{"role": "system", "content": system}, {"role": "user", "content": user}]
