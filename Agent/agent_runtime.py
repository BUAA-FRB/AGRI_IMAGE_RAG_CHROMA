import os
from typing import Any, Dict, Optional

from .io import load_final_output, get_primary_prediction, resolve_asset_path
from .strategy_engine import build_strategy_from_final_output
from .visualization import visualize_timeline, build_evidence_grid, write_dashboard_html
from .utils import new_run_id, utc_ts, ensure_dir, write_json, safe_relpath
from .qwen25_client import Qwen25Client, Qwen25GenConfig
from .refine_with_qwen25 import refine_strategy_with_qwen25


def _write_markdown_report(
    out_md: str,
    base_dir: str,
    final_obj: Dict[str, Any],
    strategy: Dict[str, Any],
    timeline_png: str,
    evidence_grid_png: str,
    montage_png: Optional[str],
    refine_json: Optional[Dict[str, Any]],
) -> None:
    s = strategy.get("summary", {})
    mf = strategy.get("model_fields", {})
    tl_rel = safe_relpath(timeline_png, base_dir)
    grid_rel = safe_relpath(evidence_grid_png, base_dir)
    montage_rel = safe_relpath(montage_png, base_dir) if montage_png else None

    lines = []
    lines.append("# 农业灾害智能体报告（灾前 / 灾中 / 灾后）\n")
    lines.append(f"- schema: `{final_obj.get('schema','')}`")
    lines.append(f"- id: `{final_obj.get('id','')}`")
    lines.append(f"- time_utc: `{final_obj.get('time_utc','')}`\n")

    lines.append("## 1. 预测摘要\n")
    lines.append(f"- label: **{s.get('pred_label','')}**")
    lines.append(f"- playbook: **{s.get('playbook_key','')}**")
    lines.append(f"- confidence: **{s.get('confidence',0):.2f}**")
    lines.append(f"- severity: **{s.get('severity','')}**")
    lines.append(f"- evidence refs: {', '.join(s.get('evidence_refs',[]) or [])}\n")

    if refine_json:
        lines.append("## 2. 策略整理（Qwen2.5-3B 可选）\n")
        lines.append(f"**一句话关键结论：** {refine_json.get('key_takeaway','')}\n")
        pr = refine_json.get("priorities", {})
        lines.append("**分阶段优先事项：**")
        lines.append(f"- 灾前：{pr.get('灾前', [])}")
        lines.append(f"- 灾中：{pr.get('灾中', [])}")
        lines.append(f"- 灾后：{pr.get('灾后', [])}\n")
        lines.append("**需要补充的信息：**")
        for x in refine_json.get("data_requests", []) or []:
            lines.append(f"- {x}")
        lines.append("")

    lines.append("## 3. 可视化\n")
    lines.append(f"![timeline]({tl_rel})\n")
    lines.append(f"![evidence_grid]({grid_rel})\n")
    if montage_rel:
        lines.append(f"![montage]({montage_rel})\n")

    lines.append("## 4. 模型解释与不确定性\n")
    lines.append(f"- conclusion: {mf.get('conclusion','')}")
    lines.append(f"- reason: {mf.get('reason','')}")
    lines.append(f"- uncertainty: {mf.get('uncertainty','')}\n")

    lines.append("## 5. 灾前 / 灾中 / 灾后行动清单\n")
    for phase in ["灾前", "灾中", "灾后"]:
        lines.append(f"### {phase}\n")
        items = strategy.get("phases", {}).get(phase, []) or []
        for i, it in enumerate(items, 1):
            lines.append(f"**{i}. {it.get('title','')}**")
            lines.append(f"- 目标：{it.get('goal','')}")
            lines.append("- 行动：")
            for a in it.get("actions", []) or []:
                lines.append(f"  - {a}")
            lines.append("- 产物：")
            for d in it.get("deliverables", []) or []:
                lines.append(f"  - {d}")
            lines.append("")
        lines.append("")

    lines.append("## 6. 建议补采信息（用于灾前预警与灾后复盘）\n")
    for x in strategy.get("data_requests", []) or []:
        lines.append(f"- {x}")
    lines.append("")

    lines.append("## 7. 安全与合规说明\n")
    for d in strategy.get("disclaimer", []) or []:
        lines.append(f"- {d}")
    lines.append("")

    ensure_dir(os.path.dirname(out_md) or ".")
    with open(out_md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


class AgriAgent:
    """
    只读输入 JSON（agri_rag.final_output.v2），生成灾前/灾中/灾后策略与可视化/报告。
    不依赖 agri_rag 其他模块。
    """
    def __init__(
        self,
        enable_qwen25_refine: bool = False,
        qwen25_model_path: str = "./models/Qwen2.5-3B",
    ) -> None:
        self.enable_qwen25_refine = bool(enable_qwen25_refine)
        self.qwen25_model_path = qwen25_model_path
        self._qwen25: Optional[Qwen25Client] = None

    def _get_qwen25(self) -> Qwen25Client:
        if self._qwen25 is None:
            self._qwen25 = Qwen25Client(self.qwen25_model_path)
        return self._qwen25

    def run_from_json(
        self,
        final_json_path: str,
        out_dir: str = "./agent_runs",
        notes: str = "",
        max_evidence: int = 6,
    ) -> Dict[str, Any]:
        fo = load_final_output(final_json_path)

        run_id = new_run_id("agent")
        run_dir = ensure_dir(os.path.join(out_dir, run_id))

        label, conf, refs, reason = get_primary_prediction(fo)
        concl = str(fo.normalized.get("conclusion") or fo.parsed_json.get("conclusion") or "")
        uncertainty = str(fo.normalized.get("uncertainty") or fo.parsed_json.get("uncertainty") or "")

        strategy = build_strategy_from_final_output(
            pred_label=label,
            confidence=conf,
            refs=refs if refs else (fo.evidence_order[:min(len(fo.evidence_order), 3)]),
            reason=reason,
            conclusion=concl,
            uncertainty=uncertainty,
            evidence=fo.evidence,
            notes=notes,
        )

        # visuals
        timeline_png = os.path.join(run_dir, "timeline.png")
        evidence_grid_png = os.path.join(run_dir, "evidence_grid.png")
        visualize_timeline(strategy, timeline_png)
        build_evidence_grid(fo.assets, fo.evidence_order, evidence_grid_png, max_evidence=max_evidence)

        montage_rel = fo.assets.get("montage", {}).get("path")
        montage_png = resolve_asset_path(fo, montage_rel) if montage_rel else None
        if montage_png and not os.path.exists(montage_png):
            montage_png = None

        # optional refine
        refine_pack = None
        refine_json = None
        if self.enable_qwen25_refine:
            q25 = self._get_qwen25()
            refine_pack = refine_strategy_with_qwen25(strategy, q25, gen=Qwen25GenConfig())
            refine_json = (refine_pack or {}).get("json")

        # markdown report + html dashboard
        report_md = os.path.join(run_dir, "report.md")
        dashboard_html = os.path.join(run_dir, "dashboard.html")
        _write_markdown_report(
            out_md=report_md,
            base_dir=run_dir,
            final_obj=read_json_passthrough(final_json_path),
            strategy=strategy,
            timeline_png=timeline_png,
            evidence_grid_png=evidence_grid_png,
            montage_png=montage_png,
            refine_json=refine_json,
        )
        write_dashboard_html(
            out_html=dashboard_html,
            base_dir=run_dir,
            title="农业灾害智能体 Dashboard",
            timeline_png=timeline_png,
            evidence_grid_png=evidence_grid_png,
            montage_png=montage_png,
            strategy=strategy,
            final_output=read_json_passthrough(final_json_path),
            refine_json=refine_json,
        )

        # agent output json
        out_json = {
            "schema": "agri_rag.agent_output.v1",
            "schema_version": "1.0",
            "time_utc": utc_ts(),
            "run_id": run_id,
            "inputs": {
                "final_json_path": os.path.abspath(final_json_path),
                "source_schema": fo.schema,
                "source_id": fo.id,
                "source_time_utc": fo.time_utc,
            },
            "prediction": {
                "label": label,
                "confidence": conf,
                "refs": strategy.get("summary", {}).get("evidence_refs", []),
            },
            "strategy": strategy,
            "visuals": {
                "timeline_png": timeline_png,
                "evidence_grid_png": evidence_grid_png,
                "montage_png": montage_png,
                "report_md": report_md,
                "dashboard_html": dashboard_html,
            },
            "refine_qwen25": refine_pack,
        }
        write_json(os.path.join(run_dir, "agent_output.json"), out_json)
        return out_json


def read_json_passthrough(path: str) -> Dict[str, Any]:
    # 避免循环 import
    from .utils import read_json
    return read_json(path)
