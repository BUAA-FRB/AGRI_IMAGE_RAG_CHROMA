import ast
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple

from .playbook import PLAYBOOKS, resolve_playbook_key
from .safety import sanitize_actions, add_global_disclaimer
from .utils import try_float


@dataclass
class EvidenceStats:
    topk: int
    consistent_count: int
    consistent_ratio: float
    min_distance: Optional[float]
    mean_distance: Optional[float]


@dataclass
class StrategySummary:
    pred_label: str
    playbook_key: str
    confidence: float
    severity: str
    evidence_refs: List[str]
    evidence_stats: Dict[str, Any]
    notes: str = ""


def _parse_labels_present(x: Any) -> List[str]:
    if x is None:
        return []
    if isinstance(x, list):
        return [str(i) for i in x]
    s = str(x)
    # labels_present 形如 '["waterway"]'
    try:
        v = ast.literal_eval(s)
        if isinstance(v, list):
            return [str(i) for i in v]
    except Exception:
        pass
    return []


def compute_evidence_stats(evidence: Dict[str, Any], refs: List[str], pred_label: str) -> EvidenceStats:
    dists: List[float] = []
    consistent = 0
    topk = 0
    for r in refs:
        e = evidence.get(r)
        if not isinstance(e, dict):
            continue
        topk += 1
        d = try_float(e.get("distance"), None)
        if d is not None:
            dists.append(float(d))
        lp = _parse_labels_present(e.get("labels_present"))
        if pred_label and pred_label in [x.lower() for x in lp]:
            consistent += 1
        else:
            # 有些 labels_present 是别的标签，但仍可能是负例证据；这里不算一致
            pass

    min_d = min(dists) if dists else None
    mean_d = (sum(dists) / len(dists)) if dists else None
    ratio = (consistent / topk) if topk > 0 else 0.0

    return EvidenceStats(
        topk=topk,
        consistent_count=consistent,
        consistent_ratio=ratio,
        min_distance=min_d,
        mean_distance=mean_d,
    )


def estimate_severity(conf: float, consistent_ratio: float) -> str:
    """
    你没有 mask/面积信息时，用“置信度 + 证据一致性”给一个保守分级：
    - 高置信 + 高一致：中（不直接给“重”，避免夸大）
    - 中置信：轻
    - 低置信或一致性差：待复核
    """
    if conf >= 0.90 and consistent_ratio >= 0.6:
        return "中"
    if conf >= 0.70:
        return "轻"
    return "待复核"


def build_data_requests(pred_label: str, uncertainty: str, evidence_stats: EvidenceStats) -> List[str]:
    req = []
    # 通用补采
    req.append("同地块多时相影像（至少2-3个时间点），用于判断是否持续/扩散/回落")
    req.append("作物类型与生育期（拔节/抽穗/灌浆等），用于解释敏感期风险")
    req.append("近期农事管理记录（灌溉/排水/播种/机械作业等），用于归因与复盘")

    # 针对性补采
    if pred_label in ["waterway"]:
        req.append("灌溉/排水设施示意或渠系走向信息（渠口/低洼点/排水口）")
        req.append("降雨与地块水位/土壤湿度相关信息（若可获得）")
    if "无" not in (uncertainty or "") and uncertainty.strip():
        req.append(f"不确定性提示：{uncertainty.strip()}（建议结合上述信息复核）")

    # 若证据一致性差，强调复核
    if evidence_stats.consistent_ratio < 0.4:
        req.append("补充更高分辨率影像或近景巡查照片，用于消除相似纹理误判")
    return req


def build_strategy_from_final_output(
    pred_label: str,
    confidence: float,
    refs: List[str],
    reason: str,
    conclusion: str,
    uncertainty: str,
    evidence: Dict[str, Any],
    notes: str = "",
) -> Dict[str, Any]:
    pb_key = resolve_playbook_key(pred_label)
    pb = PLAYBOOKS.get(pb_key, PLAYBOOKS["通用"])

    est = compute_evidence_stats(evidence=evidence, refs=refs, pred_label=pred_label.lower())
    severity = estimate_severity(confidence, est.consistent_ratio)

    # actions 安全清洗
    phases: Dict[str, List[Dict[str, Any]]] = {}
    for phase in ["灾前", "灾中", "灾后"]:
        items = []
        for it in pb.get(phase, []):
            it2 = dict(it)
            it2["actions"] = sanitize_actions(list(it2.get("actions", [])))
            items.append(it2)
        phases[phase] = items

    summary = StrategySummary(
        pred_label=pred_label,
        playbook_key=pb_key,
        confidence=float(confidence),
        severity=severity,
        evidence_refs=list(refs),
        evidence_stats=asdict(est),
        notes=notes,
    )

    strategy = {
        "schema": "agri_rag.agent.strategy.v1",
        "summary": asdict(summary),
        "model_fields": {
            "conclusion": conclusion or "",
            "reason": reason or "",
            "uncertainty": uncertainty or "",
        },
        "phases": phases,
        "data_requests": build_data_requests(pred_label=pred_label, uncertainty=uncertainty, evidence_stats=est),
    }

    return add_global_disclaimer(strategy)
