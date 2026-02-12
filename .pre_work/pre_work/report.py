from __future__ import annotations
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
import csv

def write_next30_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)

def build_report_md(site_meta: Dict[str, Any], bundle: Dict[str, Any], out_dir: Path, llm_md: Optional[str] = None) -> str:
    time_utc = datetime.now(timezone.utc).isoformat()
    next30 = bundle.get("tables", {}).get("next30") or []
    thresholds = bundle.get("thresholds", {}) or {}
    summary = bundle.get("risk", {}).get("summary") or {}

    lines: List[str] = []
    lines.append("# 气象-水文-土壤风险预警报告（灾前）")
    lines.append(f"> 生成时间：{time_utc}")
    lines.append("")

    lines.append("## 1. 关键信息概览（一页摘要）")
    lines.append(f"- 地块：{site_meta.get('name','(unknown)')}（lat={site_meta.get('lat')}, lon={site_meta.get('lon')}）")
    lines.append(f"- 作物/生育期：{site_meta.get('crop','unknown')}（{site_meta.get('stage','unknown')}）")
    for k in ["waterlogging","drought","heat","frost","disease_pressure"]:
        s = summary.get(k) or {}
        if s:
            lines.append(f"- {k}: 峰值 {float(s.get('peak',0)):.0f}（{s.get('peak_day','')}），首次 L2={s.get('first_L2') or '无'}，首次 L3={s.get('first_L3') or '无'}")
    lines.append("")

    lines.append("## 1.5 图表与量化推演")
    lines.append("### 1.5.1 多风险曲线与作业窗口")
    lines.append("![](assets/charts/risk_curves.png)")
    lines.append("### 1.5.2 降水/ET0/水分收支（趋势）")
    lines.append("![](assets/charts/water_balance.png)")
    lines.append("### 1.5.3 土壤含水量与异常（相对过去两周）")
    lines.append("![](assets/charts/soil_moisture.png)")
    lines.append("### 1.5.4 风险等级日历（L0-L3）")
    lines.append("![](assets/charts/trigger_heatmap.png)")
    lines.append("")

    lines.append("## 2. 未来 30 天风险时间表（可直接抄到台账）")
    lines.append("> 字段含义：risk_* 为 0–100；level_* 为 L0–L3；trig_* 为触发信号摘要。")
    lines.append("")
    show = next30[:14]
    if show:
        headers = list(show[0].keys())
        lines.append("| " + " | ".join(headers) + " |")
        lines.append("|" + "|".join(["---"]*len(headers)) + "|")
        for r in show:
            lines.append("| " + " | ".join([str(r.get(h,"")) for h in headers]) + " |")
    else:
        lines.append("- （无数据）")
    lines.append("")
    lines.append("完整 CSV：`risk_table_next30.csv`")
    lines.append("")

    lines.append("## 3. 风险解读与行动窗口（大模型解读）")
    if llm_md and llm_md.strip():
        lines.append(llm_md.strip())
    else:
        lines.append("> 未启用大模型解读（可用 --model_path 或移除 --no_llm）。")
    lines.append("")

    lines.append("## 4. 阈值与敏感点（为什么不等于普通天气预报）")
    lines.append("- 我们把预报转换为：3/7 日累计、ET0、水分收支、土壤湿度异常、连旱天数、VPD、径流代理等“可提前触发的风险信号”。")
    lines.append("- 阈值（可按地区/作物调参）：")
    for k, v in thresholds.items():
        lines.append(f"  - {k}: {v}")
    lines.append("")

    lines.append("## 5. 仍需补充的信息（提升预警可执行性）")
    lines.append("- 地块排水条件（沟渠/暗管/泵站能力）与低洼区分布。")
    lines.append("- 实际灌溉能力与水源约束。")
    lines.append("- 作物精确生育期/品种（热/霜阈值与敏感期强相关）。")
    lines.append("- 田间传感器（墒情/水位/雨量）或近邻站点用于回测校准。")
    lines.append("")
    return "\n".join(lines) + "\n"
