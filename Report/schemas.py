from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class KnowledgeRef:
    key: str               # e.g., "K1"
    title: str = ""
    section: str = ""
    path: str = ""
    snippet: str = ""      # short excerpt for appendix


@dataclass
class PhaseAction:
    action: str
    why: str = ""
    refs: List[str] = field(default_factory=list)


@dataclass
class AdviserPlan:
    label: str = ""
    confidence: Optional[float] = None
    risk_level: str = ""
    key_takeaway: str = ""
    phases: Dict[str, List[PhaseAction]] = field(default_factory=dict)
    loss_assessment: Dict[str, Any] = field(default_factory=dict)
    recovery_rebuild: Dict[str, Any] = field(default_factory=dict)
    data_requests: List[str] = field(default_factory=list)
    disclaimer: str = ""


@dataclass
class PredInfo:
    pred_label: str = ""
    confidence: Optional[float] = None
    reason: str = ""
    conclusion: str = ""
    uncertainty: str = ""
    refs: List[str] = field(default_factory=list)
    canon_label: str = ""
    canon_labels: List[str] = field(default_factory=list)
    pred_id: str = ""


@dataclass
class ReportContext:
    time_utc: str = ""
    input_json: str = ""
    schema: str = ""
    pred: PredInfo = field(default_factory=PredInfo)
    adviser_plan: AdviserPlan = field(default_factory=AdviserPlan)
    knowledge_refs: List[KnowledgeRef] = field(default_factory=list)
    retrieval_query: str = ""
