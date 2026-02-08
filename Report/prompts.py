from __future__ import annotations

from typing import Dict, List
from .schemas import ReportContext


_SYSTEM_PROMPT = (
    "你是一名‘农业遥感 + 农艺处置’领域的专业报告撰写助手。\n"
    "任务：根据给定的模型研判结果与专家建议，生成一份结构清晰、表达严谨、可执行性强的《智能农业研判与处置建议报告》。\n"
    "\n"
    "【输出格式硬性规范（必须遵守）】\n"
    "1) 只输出纯 Markdown 正文：禁止输出任何代码围栏/代码块（包括 ```、```markdown、```md 等）。\n"
    "2) 报告必须以单个 H1 标题开头（# ...），紧接一行引用块写生成时间。\n"
    "3) 列表风格必须规范且一致：\n"
    "   - 有序列表用“1. 2. 3.”；无序列表用“- ”。\n"
    "   - 严禁出现 “- 1.” “- 2.” 这类混合写法。\n"
    "4) 不要在多个章节重复粘贴同一段‘研判结论’；各章内容要各司其职。\n"
    "5) 不要原样复述检索 Query；不要大段粘贴证据原文；只做摘要并按需引用 [K1]..[K6]。\n"
    "6) 语言风格：专业、克制、可执行，少形容词，多检查要点/记录要点/完成标准。\n"
    "7) 合规：报告仅供决策辅助，不替代现场踏查与当地农技规范。\n"
)


def build_messages(ctx: ReportContext) -> List[Dict[str, str]]:
    """Build chat messages for Qwen2.5 Instruct."""

    # Knowledge appendix mapping: K1..K6
    k_lines: List[str] = []
    for kr in ctx.knowledge_refs:
        snippet = (kr.snippet or "").strip().replace("\n", " ")
        if len(snippet) > 260:
            snippet = snippet[:260] + "…"
        k_lines.append(
            f"- [{kr.key}] {kr.title}｜{kr.section}｜{kr.path}\n  - 摘要：{snippet}"
        )
    knowledge_block = "\n".join(k_lines) if k_lines else "- (无)"

    # Phase actions: provide as ordered lists WITHOUT bullet prefix to avoid priming "- 1."
    phases = ctx.adviser_plan.phases or {}
    phase_names = ["灾前", "灾中", "灾后"]
    phase_block_lines: List[str] = []
    for pn in phase_names:
        items = phases.get(pn, []) or []
        phase_block_lines.append(f"#### {pn}")
        if not items:
            phase_block_lines.append("- （无）")
        else:
            for i, it in enumerate(items, 1):
                refs = ""
                if getattr(it, "refs", None):
                    # it.refs is already like ["K1","K2"] in your upstream plan
                    refs = f"（参考：{', '.join(it.refs)}）"
                why = (it.why or "").strip()
                if why:
                    phase_block_lines.append(f"{i}. {it.action}：{why}{refs}")
                else:
                    phase_block_lines.append(f"{i}. {it.action}{refs}")
    phase_block = "\n".join(phase_block_lines)

    pred = ctx.pred
    plan = ctx.adviser_plan

    # Strongly structured user prompt + explicit ban on code fences and mixed list style
    user_prompt = f"""请根据以下“结构化信息”生成一份高质量 Markdown 报告。请严格遵守下面规则：

【硬性规则】
- 只输出纯 Markdown 正文：禁止输出任何代码围栏/代码块（包括 ```、```markdown、```md 等）。
- 列表格式必须规范：有序列表仅用“1. 2. 3.”；无序列表仅用“- ”；严禁出现“- 1.”这种混排。
- 不要原样复述检索 Query；不要大段粘贴证据文本；只做摘要并在关键处引用 [K1]..[K6]。
- 不要在多个章节重复粘贴同一段研判结论；各章节要分别回答不同问题。
- 行动建议必须“可执行”：尽量写清楚【优先级/目的/怎么做/记录什么/完成标准】。
- 输出必须包含明确章节结构，且开头必须是单个 H1 标题 + 生成时间引用块。

【建议结构（请按此输出）】
# 智能农业研判与处置建议报告
> 生成时间：{ctx.time_utc}

## 1. 关键信息概览（一页摘要）
- 核心结论（≤30字）
- 风险等级与主要风险点（2-4条）
- 立即行动（24-72h，3-6条，含优先级 P0/P1）
- 需补充信息（2-4条）

## 2. 研判结论与证据链
- 研判结论（可复述给现场人员）
- 主要依据（reason 的凝练版）
- 不确定性与误判风险（以及如何验证）
- 证据引用（按需引用 [Kx]）

## 3. 风险解读（为什么要管）
- 影响路径（1-3条）
- 现场可观察迹象（3-5条）
- 验证建议（2-3条：怎么排除混淆）

## 4. 分阶段行动方案（可直接执行）
> 优先级：P0 立即处理｜P1 1-3天｜P2 1-2周
分别写：灾前 / 灾中 / 灾后
每条建议尽量包含：目的、怎么做、记录项、完成标准，并尽量引用 [Kx]。

## 5. 损失评估与取证要点
- 需要量化的指标
- 如何记录（表格/照片/位置等）
- 输出物清单
- 注意事项（避免偏差）

## 6. 恢复重建与监测计划
- 恢复重建优先序（1-5条）
- 监测计划（频次/方法/阈值或告警条件）
- 复盘要点（面向下一次改进）

## 7. 仍需补充的信息
把 data_requests 变成“为什么重要/如何获取”的列表。

## 8. 免责声明
给出合规与安全提示，语气克制。

## 9. 知识依据（摘要）
按需列出 [K1]..[K6]，每条一行，避免长段引用。

—— 结构化信息如下 ——


### A) 任务元信息
- time_utc: {ctx.time_utc}
- input_json: {ctx.input_json}
- schema: {ctx.schema}

### B) 模型研判（pred）
- 预测标签: {pred.pred_label}
- 规范化标签: {pred.canon_label}
- 置信度: {pred.confidence}
- 证据引用: {', '.join(pred.refs) if pred.refs else '(无)'}
- 原因（reason）: {pred.reason}
- 结论（conclusion）: {pred.conclusion}
- 不确定性（uncertainty）: {pred.uncertainty}

### C) 专家建议结构（adviser_plan）
- label: {plan.label}
- confidence: {plan.confidence}
- risk_level: {plan.risk_level}
- key_takeaway: {plan.key_takeaway}

#### C.1 分阶段行动清单（结构化输入）
{phase_block}

#### C.2 损失评估（loss_assessment）
{plan.loss_assessment}

#### C.3 恢复重建（recovery_rebuild）
{plan.recovery_rebuild}

#### C.4 仍需补充的信息（data_requests）
{plan.data_requests}

#### C.5 免责声明（disclaimer）
{plan.disclaimer}

### D) 检索到的知识依据（映射为 [K1]..[K6]）
{knowledge_block}

### E) 检索 Query（只用于理解上下文，禁止原样复述）
{ctx.retrieval_query}
"""

    return [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
