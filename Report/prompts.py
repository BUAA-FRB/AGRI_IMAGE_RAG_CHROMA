from __future__ import annotations

from typing import Dict, List
from .schemas import ReportContext


_SYSTEM_PROMPT = (
    "你是一名‘农业遥感 + 农艺处置’领域的专业顾问，写作对象是现场处置与管理人员。\n"
    "你的报告更像“交接单 + 处置建议”，要求清楚、克制、可执行：\n"
    "把事情讲明白：现在看到什么 → 担心什么 → 先做什么 → 怎么验证 → 怎么记录与复盘。\n"
    "\n"
    "【输出格式硬性规范（必须遵守）】\n"
    "1) 只输出纯 Markdown 正文：禁止输出任何代码围栏/代码块（包括 ```、```markdown、```md 等）。\n"
    "2) 报告必须以单个 H1 标题开头（# ...），紧接一行引用块写生成时间。\n"
    "3) 列表风格必须规范且一致：\n"
    "   - 有序列表用“1. 2. 3.”；无序列表用“- ”。\n"
    "   - 严禁出现 “- 1.” “- 2.” 这类混合写法。\n"
    "4) 不要在多个章节重复粘贴同一段‘研判结论’；各章内容要各司其职。\n"
    "5) 不要原样复述检索 Query；不要大段粘贴证据原文；只做摘要并按需引用 [K1]..[K6]。\n"
    "\n"
    "【写作风格（降低 AI 味，必须执行）】\n"
    "- 段落短：1–3 句一段；关键内容用要点列出。\n"
    "- 用自然衔接：例如“因此/所以/现场建议/优先/注意/可以这样核验”。\n"
    "- 少空话套话：避免“高度关注/意义重大/非常关键”等泛化表达，用具体动作替代。\n"
    "- 不确定性要说清楚：给出如何核验、如何排除混淆。\n"
    "\n"
    "【合规边界】\n"
    "报告仅供决策辅助，不替代现场踏查与当地农技规范；相关边界与安全提示请融入“现场核验与记录模板/注意事项”，不要单独写‘免责声明’章节。\n"
)


def build_messages(ctx: ReportContext) -> List[Dict[str, str]]:
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

    # Phase actions: ordered lists WITHOUT bullet prefix (avoid priming "- 1.")
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
                    refs = f"（参考：{', '.join(it.refs)}）"
                why = (it.why or "").strip()
                if why:
                    phase_block_lines.append(f"{i}. {it.action}：{why}{refs}")
                else:
                    phase_block_lines.append(f"{i}. {it.action}{refs}")
    phase_block = "\n".join(phase_block_lines)

    pred = ctx.pred
    plan = ctx.adviser_plan

    user_prompt = f"""请根据以下“结构化信息”生成一份高质量 Markdown 报告。请严格遵守规则。

【硬性规则】
- 只输出纯 Markdown 正文：禁止输出任何代码围栏/代码块（包括 ```、```markdown、```md 等）。
- 列表格式必须规范：有序列表仅用“1. 2. 3.”；无序列表仅用“- ”；严禁出现“- 1.”混排。
- 不要原样复述检索 Query；不要大段粘贴证据文本；只做摘要并在关键处引用 [K1]..[K6]。
- 不要在多个章节重复同一段结论；同一事实可换说法，但不要复制粘贴。
- 行动建议必须“可执行”：尽量写清楚【优先级/目的/怎么做/记录什么/完成标准】。
- 报告开头必须是单个 H1 标题 + 生成时间引用块。

【风格要求（减少 AI 味）】
- 像人类专家给现场写交接：短段落、自然过渡、少形容词、多要点。
- 避免口号式句子；用具体检查点、记录字段、完成标准来表达专业性。

【建议结构（请按此输出）】
# 智能农业研判与处置建议报告
> 生成时间：{ctx.time_utc}

## 1. 关键信息概览（一页摘要）
- 核心结论（≤30字）
- 风险等级与主要风险点（2-4条）
- 立即行动（24-72h，3-6条，含优先级 P0/P1）
- 需补充信息（2-4条）

## 2. 研判结论与证据链
用一段自然语言复述结论（像对现场人员说），再用要点列“依据/混淆项/如何核验”，按需引用 [Kx]。

## 3. 风险解读（为什么要管）
- 影响路径（1-3条）
- 现场可观察迹象（3-5条）
- 验证建议（2-3条：怎么排除混淆）

## 4. 分阶段行动方案（可直接执行）
> 优先级：P0 立即处理｜P1 1-3天｜P2 1-2周
分别写：灾前 / 灾中 / 灾后（每条建议尽量包含：目的、怎么做、记录项、完成标准，并尽量引用 [Kx]）。

## 5. 损失评估与取证要点
- 指标
- 记录方式（表格/照片/位置/时间）
- 输出物清单
- 注意事项（避免偏差）

## 6. 恢复重建与监测计划
- 恢复重建优先序（1-5条）
- 监测计划（频次/方法/阈值或告警条件）
- 复盘要点（面向下一次改进）

## 7. 仍需补充的信息
把 data_requests 改写成“为什么重要/如何获取”。

## 8. 现场核验与记录模板（可直接抄用）
请给出一份“现场核验步骤 + 记录表字段”的模板，方便直接落地执行：
- 核验顺序（例如：上游→主渠→支渠→田间末端 / 设施→水位→出入水口等）
- 每一步需要观察什么、如何判断“堵/漏/倒灌/异常水位”
- 建议记录字段（时间、位置、照片编号、现象、初判原因、处置动作、复测结果、责任人）
- 留痕建议（前后对比、关键点位拍摄）
- 安全与边界提示（融入本节，不要写成免责声明标题）

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

#### C.5 安全与边界提示（来源：disclaimer 字段，融入第 8 节，不要单独写免责声明）
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
