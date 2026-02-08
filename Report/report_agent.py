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
# Markdown post-processing (NEW, for stability)
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

    # Case 1: direct full-match fence wrapper
    m = re.match(r"^\s*```(?:markdown|md)?\s*\n([\s\S]*?)\n```\s*$", t, flags=re.IGNORECASE)
    if m:
        return m.group(1).strip()

    # Case 2: exactly two fence lines, first at start and last at end
    fence_lines = list(_FENCE_LINE_RE.finditer(t))
    if len(fence_lines) == 2:
        first = fence_lines[0]
        last = fence_lines[1]
        # first fence must start at 0 after stripping, last fence must end at end
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
    if not t:
        title_time = ctx.time_utc or datetime.now(timezone.utc).isoformat()
        return f"# 智能农业研判与处置建议报告\n\n> 生成时间：{title_time}\n"

    lines = t.splitlines()
    # find first non-empty line
    i = 0
    while i < len(lines) and not lines[i].strip():
        i += 1

    title_time = ctx.time_utc or datetime.now(timezone.utc).isoformat()

    if i >= len(lines) or not lines[i].lstrip().startswith("# "):
        # Prepend title + time
        prefix = f"# 智能农业研判与处置建议报告\n\n> 生成时间：{title_time}\n\n"
        return prefix + t

    # ensure time block exists in first few lines (not strict, but helpful)
    head = "\n".join(lines[i:i+6])
    if "生成时间" not in head:
        # insert after title line
        title_line = lines[i]
        rest = lines[i+1:]
        new = [title_line, "", f"> 生成时间：{title_time}", ""]
        new.extend(rest)
        # keep any leading blank lines removed by lstrip
        return "\n".join(lines[:i] + new).lstrip() + "\n"

    return t


def _postprocess_markdown(md: str, ctx: ReportContext) -> str:
    md2 = _strip_outer_markdown_fence(md)
    md2 = _normalize_mixed_numbered_bullets(md2)
    md2 = _ensure_title_and_time(md2, ctx)
    return md2.strip() + "\n"


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

        # Postprocess for strict formatting guarantees
        md = _postprocess_markdown(md, ctx)
        return md

    def save_report(self, advice_json_path: str, out_dir: str) -> str:
        from .io_utils import write_text

        out_dir_p = Path(out_dir)
        out_dir_p.mkdir(parents=True, exist_ok=True)

        pkg = read_json(advice_json_path)
        ctx = _build_context(pkg)

        pred_id = ctx.pred.pred_id or "unknown"
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        out_path = out_dir_p / f"report_{pred_id}_{ts}.md"

        md = self.generate_report_markdown(advice_json_path)
        write_text(out_path, md)
        return str(out_path)
