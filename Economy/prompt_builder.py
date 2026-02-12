from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from .models import EconomyResult


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def _fmt_ratio(x: Any) -> str:
    v = _safe_float(x, 0.0)
    return f"{v * 100:.2f}%"


def _fmt_money(x: Any, currency: str) -> str:
    if x is None:
        return "N/A"
    try:
        v = float(x)
    except Exception:
        return "N/A"
    return f"{currency} {v:,.2f}"


def _compact_json(obj: Any, max_chars: int = 8000) -> str:
    """
    NOTE: This is for input prompt (not LLM output). The output side forbids code blocks.
    """
    s = json.dumps(obj, ensure_ascii=False, indent=2)
    if len(s) <= max_chars:
        return s
    return s[:max_chars] + "\n...<truncated>...\n"


def _build_items_table(econ: EconomyResult) -> str:
    """
    Provide a compact, human-readable table to reduce the chance the model copies raw JSON.
    """
    lines: List[str] = []
    lines.append("| label | conf | 受灾比例 | 减产P10/P50/P90 | 损失P10/P50/P90 |")
    lines.append("|---|---:|---:|---:|---:|")
    for it in econ.items:
        yl = f"{_fmt_ratio(it.yield_loss_frac_p10)}/{_fmt_ratio(it.yield_loss_frac_p50)}/{_fmt_ratio(it.yield_loss_frac_p90)}"
        lv = f"{_fmt_money(it.loss_value_p10, econ.currency)}/{_fmt_money(it.loss_value_p50, econ.currency)}/{_fmt_money(it.loss_value_p90, econ.currency)}"
        lines.append(f"| {it.label} | {it.confidence:.2f} | {_fmt_ratio(it.affected_ratio)} | {yl} | {lv} |")
    return "\n".join(lines)


def _build_forecast_table(charts_meta: Dict[str, Any], max_rows: int = 8) -> str:
    month_labels = charts_meta.get("month_labels") or []
    series = charts_meta.get("series") or {}
    base = series.get("baseline") or {}
    mit = series.get("mitigated") or {}
    base_p10 = base.get("p10") or []
    base_p50 = base.get("p50") or []
    base_p90 = base.get("p90") or []
    mit_p50 = mit.get("p50") or []

    n = min(len(month_labels), len(base_p10), len(base_p50), len(base_p90), len(mit_p50), max_rows)
    lines: List[str] = []
    lines.append("| 月份 | 基线P50 | 基线区间(P10–P90) | 处置后P50(情景) |")
    lines.append("|---|---:|---:|---:|")
    for i in range(n):
        lines.append(
            f"| {month_labels[i]} | {base_p50[i]:,.2f} | {base_p10[i]:,.2f}–{base_p90[i]:,.2f} | {mit_p50[i]:,.2f} |"
        )
    return "\n".join(lines)


def build_llm_prompt(
    latest: Dict[str, Any],
    econ: EconomyResult,
    charts_meta: Dict[str, Any],
) -> Dict[str, str]:
    """
    LLM 输出只负责：经济研判“第 3 节正文”，不允许输出整篇报告（避免标题/编号冲突）。
    并严格禁止 fenced code block / 混排列表 / 复制粘贴输入 JSON。
    """
    normalized = latest.get("normalized") or {}
    conclusion = normalized.get("conclusion") or (latest.get("parsed_json") or {}).get("conclusion") or ""
    uncertainty = normalized.get("uncertainty") or (latest.get("parsed_json") or {}).get("uncertainty") or ""

    preds = normalized.get("predicted_labels") or (latest.get("parsed_json") or {}).get("predicted_labels") or []
    preds_simple = [{"label": p.get("label"), "confidence": p.get("confidence"), "reason": p.get("reason")} for p in preds]

    series = charts_meta.get("series") or {}
    mitigation_reduction = _safe_float(series.get("mitigation_reduction"), 0.0)
    tau_months = _safe_float(series.get("tau_months"), 0.0)

    unit = econ.currency if econ.area_mode == "gsd" else f"{econ.currency}/1{econ.normalize_to}"

    # ===== System Prompt (format hard rules) =====
    system = (
        "你是一名农业经济损失评估与风险管理专家。你必须严格使用输入数据进行分析，不得虚构数字。\n"
        "\n"
        "【输出格式硬性规范（必须遵守）】\n"
        "1) 只输出纯 Markdown 正文：禁止输出任何 fenced code block（例如以三个反引号开头/结尾的代码围栏），也不要输出‘markdown’字样的代码块声明。\n"
        "2) 你只输出“报告第 3 节：经济研判结论（大模型分析）”的正文内容，不要输出整篇报告：\n"
        "   - 禁止输出 H1 标题（# ...）\n"
        "   - 禁止输出 H2 标题（## ...）\n"
        "   - 允许使用 H3（### ...）作为小节标题。\n"
        "3) 列表风格必须规范且一致：\n"
        "   - 有序列表只用 ‘1. 2. 3.’\n"
        "   - 无序列表只用 ‘- ’\n"
        "   - 严禁出现 ‘- 1.’ 这类混排。\n"
        "4) 不要复制粘贴输入中的 JSON/长表格；引用数字时，用简短方式给出并注明字段名。\n"
        "\n"
        "【写作风格】\n"
        "- 像写给现场管理者的经济分析交接：短段落、少口号、多可执行要点。\n"
        "- 关注经济逻辑与止损机制：受灾比例→减产→产值损失；处置为什么能降低损失。\n"
        "- 不确定性要说清楚：指出敏感参数、给出核验与补数建议。\n"
    )

    # ===== User Prompt: require section-only output =====
    # 重点：要求每个金额/比例后面带字段引用（field: ...），从而“可审计+不瞎编”
    user = f"""请基于以下输入，为《经济损失估计与处置建议报告（经济视角）》生成“第 3 节：经济研判结论（大模型分析）”的正文内容。

【再次强调：你只输出第 3 节正文，不要输出任何 # / ## 标题，不要输出整篇报告，不要输出代码围栏。】

【输入摘要】
- 灾害研判结论：{conclusion}
- 不确定性提示：{uncertainty}
- predicted_labels（摘要）：{json.dumps(preds_simple, ensure_ascii=False)}

【经济估计总览（请引用字段，不要改写成你自编的新数字）】
- 总损失：P10={_fmt_money(econ.total_loss_value_p10, econ.currency)}（field: total_loss_value_p10），
  P50={_fmt_money(econ.total_loss_value_p50, econ.currency)}（field: total_loss_value_p50），
  P90={_fmt_money(econ.total_loss_value_p90, econ.currency)}（field: total_loss_value_p90）
- 总减产比例：P10={_fmt_ratio(econ.total_yield_loss_frac_p10)}（field: total_yield_loss_frac_p10），
  P50={_fmt_ratio(econ.total_yield_loss_frac_p50)}（field: total_yield_loss_frac_p50），
  P90={_fmt_ratio(econ.total_yield_loss_frac_p90)}（field: total_yield_loss_frac_p90）
- 面积口径：{econ.area_mode}（field: area_mode），gsd={econ.gsd_m_per_px}（field: gsd_m_per_px），单位口径：{unit}
- 处置情景假设：最终损失可降低约 {mitigation_reduction*100:.1f}%（field: mitigation_reduction），损失兑现节奏参数 tau≈{tau_months:.2f} 月（field: tau_months）

【分项损失表（输入给你理解，不要整段照抄到输出；最多引用其中 2-4 个关键项）】
{_build_items_table(econ)}

【未来数月累计损失（用于解释趋势；最多引用 3-5 行关键月份）】
{_build_forecast_table(charts_meta, max_rows=6)}

【你要输出的第 3 节正文结构（必须按此结构，且只用 ### 作为小节标题）】

### 3.1 核心判断（≤6条要点）
- 用 4–6 条要点说清楚“损失规模/主要驱动/是否需要立刻止损/最敏感的不确定性”，每条要点中出现的金额或比例必须带（field: xxx）。

### 3.2 口径与计算链（写清楚，不要公式堆砌）
- 用 3–6 行解释：受灾比例如何映射到减产，再到产值损失；强调“总损失用合并减产避免重复计数”（field: combine_method 或 assumptions 中的描述）。
- 明确本次口径的局限：gsd/面积、价格与单产等（field: assumptions / warnings）。

### 3.3 损失驱动因素与敏感性（3–5条）
- 以“驱动因素→为何敏感→怎么核验/补数”的方式写。
- 至少覆盖：面积换算、价格/单产、灾害强度/持续时间、标签置信度与 tile 代理风险（如存在）。

### 3.4 经济视角的止损建议（按 P0/P1/P2）
- P0（24–72h）：3–5 条，强调“低成本高收益”的止损动作与记录字段（不要写成纯农技细节）。
- P1（1–2周）：2–4 条，强调验证、抽样与小范围投入的收益评估。
- P2（长期）：2–4 条，强调制度化台账/监测/阈值与复盘。

### 3.5 仍需补充的数据（按优先级）
- 列 4–8 条，每条写清楚：为什么重要、如何获取、能把不确定性缩小到哪里（不要写空话）。
"""

    return {"system": system, "user": user}
