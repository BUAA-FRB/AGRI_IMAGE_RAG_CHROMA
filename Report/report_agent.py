from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from .io_utils import read_json, safe_str, to_float
from .schemas import AdviserPlan, KnowledgeRef, PhaseAction, PredInfo, ReportContext
from .prompts import build_messages
from .qwen_runner import QwenRunner, QwenGenConfig

# NEW: charts
from .chart_utils import generate_charts, build_charts_markdown_section


def _parse_adviser_plan(output_obj: Dict[str, Any]) -> AdviserPlan:
    gen = (output_obj or {}).get("generation") or {}
    parsed = None

    if gen.get("parse_ok") and isinstance(gen.get("parsed_json"), dict):
        parsed = gen.get("parsed_json")
    else:
        raw = gen.get("raw_text")
        if isinstance(raw, str) and raw.strip():
            try:
                parsed = json.loads(raw)
            except Exception:
                parsed = None

    plan = AdviserPlan()
    if not isinstance(parsed, dict):
        return plan

    summary = parsed.get("summary") or {}
    plan.label = safe_str(summary.get("label"))
    plan.confidence = to_float(summary.get("confidence"))
    plan.risk_level = safe_str(summary.get("risk_level"))
    plan.key_takeaway = safe_str(summary.get("key_takeaway"))

    phases = parsed.get("phases") or {}
    if isinstance(phases, dict):
        for k, arr in phases.items():
            actions: List[PhaseAction] = []
            if isinstance(arr, list):
                for it in arr:
                    if not isinstance(it, dict):
                        continue
                    actions.append(
                        PhaseAction(
                            action=safe_str(it.get("action")),
                            why=safe_str(it.get("why")),
                            refs=list(it.get("refs") or []),
                        )
                    )
            plan.phases[safe_str(k)] = actions

    la = parsed.get("loss_assessment")
    if isinstance(la, dict):
        plan.loss_assessment = la

    rr = parsed.get("recovery_rebuild")
    if isinstance(rr, dict):
        plan.recovery_rebuild = rr

    dr = parsed.get("data_requests")
    if isinstance(dr, list):
        plan.data_requests = [safe_str(x) for x in dr if safe_str(x)]

    plan.disclaimer = safe_str(parsed.get("disclaimer"))
    return plan


def _parse_pred(output_obj: Dict[str, Any]) -> PredInfo:
    pred_obj = (output_obj or {}).get("pred") or {}
    p = PredInfo()
    if not isinstance(pred_obj, dict):
        return p

    p.pred_label = safe_str(pred_obj.get("pred_label"))
    p.confidence = to_float(pred_obj.get("confidence"))
    p.reason = safe_str(pred_obj.get("reason"))
    p.conclusion = safe_str(pred_obj.get("conclusion"))
    p.uncertainty = safe_str(pred_obj.get("uncertainty"))
    p.refs = list(pred_obj.get("refs") or [])
    p.canon_label = safe_str(pred_obj.get("canon_label"))
    p.canon_labels = list(pred_obj.get("canon_labels") or [])
    p.pred_id = safe_str(pred_obj.get("id"))
    return p


def _map_knowledge_refs(output_obj: Dict[str, Any], max_k: int = 6) -> List[KnowledgeRef]:
    ret = (output_obj or {}).get("retrieval") or {}
    hits = ret.get("hits") or []
    refs: List[KnowledgeRef] = []

    if not isinstance(hits, list):
        return refs

    for i, h in enumerate(hits[:max_k], start=1):
        if not isinstance(h, dict):
            continue
        meta = h.get("meta") or {}
        refs.append(
            KnowledgeRef(
                key=f"K{i}",
                title=safe_str(meta.get("title")),
                section=safe_str(meta.get("section")),
                path=safe_str(meta.get("path")),
                snippet=safe_str(h.get("text")),
            )
        )
    return refs


def _build_context(pkg: Dict[str, Any]) -> ReportContext:
    ctx = ReportContext()
    ctx.time_utc = safe_str(pkg.get("time_utc"))
    ctx.input_json = safe_str(pkg.get("input_json"))
    ctx.schema = safe_str(pkg.get("schema"))

    output_obj = pkg.get("output") or {}
    if isinstance(output_obj, dict):
        ctx.pred = _parse_pred(output_obj)
        ctx.adviser_plan = _parse_adviser_plan(output_obj)
        ctx.knowledge_refs = _map_knowledge_refs(output_obj, max_k=6)
        ret = output_obj.get("retrieval") or {}
        if isinstance(ret, dict):
            ctx.retrieval_query = safe_str(ret.get("query"))

    return ctx


# -------------------------
# Markdown post-processing (for stability)
# -------------------------

_FENCE_LINE_RE = re.compile(r"(?m)^\s*```.*\s*$")


def _strip_outer_markdown_fence(text: str) -> str:
    """
    Remove outermost fenced block if it wraps the whole document, e.g.
    ```markdown
    # ...
    ```
    """
    if not text:
        return text

    t = text.strip()

    m = re.match(r"^\s*```(?:markdown|md)?\s*\n([\s\S]*?)\n```\s*$", t, flags=re.IGNORECASE)
    if m:
        return m.group(1).strip()

    fence_lines = list(_FENCE_LINE_RE.finditer(t))
    if len(fence_lines) == 2:
        first = fence_lines[0]
        last = fence_lines[1]
        if first.start() == 0 and last.end() == len(t):
            inner = t[first.end(): last.start()]
            return inner.strip()

    return t


def _normalize_mixed_numbered_bullets(text: str) -> str:
    """
    Convert '- 1. xxx' / '* 2. xxx' / '+ 3. xxx' to '1. xxx'
    """
    if not text:
        return text
    return re.sub(r"(?m)^\s*[-*+]\s+(\d+)\.\s+", r"\1. ", text)


def _ensure_title_and_time(md: str, ctx: ReportContext) -> str:
    """
    Ensure the report starts with a H1 title and a generation-time quote line.
    """
    t = (md or "").lstrip()
    title_time = ctx.time_utc or datetime.now(timezone.utc).isoformat()

    if not t:
        return f"# 智能农业研判与处置建议报告\n\n> 生成时间：{title_time}\n"

    lines = t.splitlines()
    i = 0
    while i < len(lines) and not lines[i].strip():
        i += 1

    if i >= len(lines) or not lines[i].lstrip().startswith("# "):
        prefix = f"# 智能农业研判与处置建议报告\n\n> 生成时间：{title_time}\n\n"
        return prefix + t

    head = "\n".join(lines[i:i + 6])
    if "生成时间" not in head:
        title_line = lines[i]
        rest = lines[i + 1:]
        new = [title_line, "", f"> 生成时间：{title_time}", ""]
        new.extend(rest)
        return "\n".join(lines[:i] + new).lstrip() + "\n"

    return t


def _postprocess_markdown(md: str, ctx: ReportContext) -> str:
    md2 = _strip_outer_markdown_fence(md)
    md2 = _normalize_mixed_numbered_bullets(md2)
    md2 = _ensure_title_and_time(md2, ctx)
    return md2.strip() + "\n"


def _insert_charts_section(md: str, charts_md: str) -> str:
    """
    Insert charts section between Section 1 and 2 if possible,
    to avoid all-text or all-figures crowded together.
    """
    if not charts_md.strip():
        return md

    # Prefer inserting before "## 2."
    m = re.search(r"(?m)^\s*##\s+2\.", md)
    if m:
        idx = m.start()
        return md[:idx].rstrip() + "\n\n" + charts_md.strip() + "\n\n" + md[idx:].lstrip()

    # Fallback: insert before any "## 2"
    m2 = re.search(r"(?m)^\s*##\s+2\s", md)
    if m2:
        idx = m2.start()
        return md[:idx].rstrip() + "\n\n" + charts_md.strip() + "\n\n" + md[idx:].lstrip()

    # Fallback: append at end
    return md.rstrip() + "\n\n" + charts_md.strip() + "\n"


@dataclass
class ReportAgentConfig:
    model_path: str
    device: str = "auto"  # auto|cpu|cuda|mps
    dtype: str = "auto"   # auto|float16|bfloat16|float32
    max_new_tokens: int = 1600
    temperature: float = 0.4
    top_p: float = 0.85
    repetition_penalty: float = 1.05
    seed: Optional[int] = 42

    # charts
    charts_enabled: bool = True
    charts_days: int = 14


class ReportAgent:
    def __init__(self, cfg: ReportAgentConfig) -> None:
        self.cfg = cfg
        self.runner = QwenRunner(
            model_path=cfg.model_path,
            device=cfg.device,
            dtype=cfg.dtype,
        )

    def generate_report_markdown(self, advice_json_path: str) -> str:
        pkg = read_json(advice_json_path)
        ctx = _build_context(pkg)

        messages = build_messages(ctx)
        gen_cfg = QwenGenConfig(
            max_new_tokens=self.cfg.max_new_tokens,
            temperature=self.cfg.temperature,
            top_p=self.cfg.top_p,
            repetition_penalty=self.cfg.repetition_penalty,
            do_sample=True,
            seed=self.cfg.seed,
        )
        md = self.runner.generate(messages, gen_cfg)
        return _postprocess_markdown(md, ctx)

    def save_report(self, advice_json_path: str, out_dir: str) -> str:
        from .io_utils import write_text

        out_dir_p = Path(out_dir)
        out_dir_p.mkdir(parents=True, exist_ok=True)

        pkg = read_json(advice_json_path)
        ctx = _build_context(pkg)

        pred_id = ctx.pred.pred_id or "unknown"
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        run_id = f"{pred_id}_{ts}"

        out_path = out_dir_p / f"report_{run_id}.md"
        md = self.generate_report_markdown(advice_json_path)

        # Charts: generate & insert
        if self.cfg.charts_enabled:
            try:
                phases = ctx.adviser_plan.phases or {}
                bundle = generate_charts(
                    out_dir_p=out_dir_p,
                    run_id=run_id,
                    time_utc=ctx.time_utc,
                    canon_label=(ctx.pred.canon_label or ctx.pred.pred_label),
                    risk_level=ctx.adviser_plan.risk_level,
                    confidence=ctx.pred.confidence,
                    phases=phases,
                    days=self.cfg.charts_days,
                )
                charts_md = build_charts_markdown_section(bundle)
                md = _insert_charts_section(md, charts_md)
            except Exception as e:
                # Do not crash the report generation
                fallback = (
                    "## 1.5 图表与量化推演（未生成）\n"
                    "> 说明：本次未能生成图表，但不影响正文报告。\n\n"
                    f"- 错误：`{type(e).__name__}: {e}`\n"
                    "- 建议：请确认已安装依赖 `pip install matplotlib numpy`，并确保 Matplotlib 可正常调用系统字体。\n"
                )
                md = _insert_charts_section(md, fallback)

        write_text(out_path, md)
        return str(out_path)
