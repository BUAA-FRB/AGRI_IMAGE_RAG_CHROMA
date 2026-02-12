from typing import Any, Dict, List, Optional


def _as_list(x):
    return x if isinstance(x, list) else ([] if x is None else [x])


def _get_first_pred(norm: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    pred_list = norm.get("predicted_labels")
    if isinstance(pred_list, list) and pred_list:
        top = pred_list[0]
        return top if isinstance(top, dict) else None
    return None


def extract_prediction_fields(final_output: Dict[str, Any]) -> Dict[str, Any]:
    """
    兼容：
    - final_output["normalized"] (你现在的主格式)
    - fallback: final_output["parsed_json"] / final_output["raw_text"]（如果 normalized 缺失）
    """
    norm = final_output.get("normalized") or {}

    if not isinstance(norm, dict) or not norm:
        # fallback：尝试用 parsed_json
        pj = final_output.get("parsed_json")
        if isinstance(pj, dict):
            norm = {
                "predicted_labels": pj.get("predicted_labels") or [],
                "labels": _as_list(pj.get("predicted_labels", [{}])[0].get("label") if isinstance(pj.get("predicted_labels"), list) and pj.get("predicted_labels") else []),
                "conclusion": pj.get("conclusion") or "",
                "uncertainty": pj.get("uncertainty") or "",
            }

    labels = norm.get("labels") or []
    top = _get_first_pred(norm) or {}

    pred_label = str(top.get("label") or "")
    confidence = 0.0
    try:
        confidence = float(top.get("confidence") or 0.0)
    except Exception:
        confidence = 0.0
    refs: List[str] = [str(x) for x in _as_list(top.get("refs") or []) if str(x).strip()]
    reason = str(top.get("reason") or "")

    if not pred_label and labels:
        pred_label = str(labels[0])

    conclusion = str(norm.get("conclusion") or "")
    uncertainty = str(norm.get("uncertainty") or "")

    return {
        "pred_label": pred_label,
        "labels": labels,
        "confidence": confidence,
        "refs": refs,
        "reason": reason,
        "conclusion": conclusion,
        "uncertainty": uncertainty,
        "schema": str(final_output.get("schema") or ""),
        "id": str(final_output.get("id") or ""),
        "time_utc": str(final_output.get("time_utc") or ""),
    }


def extract_evidence_hint(final_output: Dict[str, Any], max_eids: int = 6) -> Dict[str, Any]:
    """
    兼容：
    - final_output["evidence_order"] + final_output["evidence"]
    - 或 evidence_context 内的 evidence_order/evidence
    """
    ev_order = final_output.get("evidence_order")
    evidence = final_output.get("evidence")

    if not ev_order or not evidence:
        ctx = final_output.get("evidence_context") or {}
        if isinstance(ctx, dict):
            ev_order = ev_order or ctx.get("evidence_order")
            evidence = evidence or ctx.get("evidence")

    ev_order = ev_order or []
    evidence = evidence or {}

    out = []
    if isinstance(ev_order, list) and isinstance(evidence, dict):
        for eid in ev_order[:max_eids]:
            e = evidence.get(eid) or {}
            out.append({
                "eid": str(eid),
                "type": e.get("type") or "",
                "distance": e.get("distance"),
                "labels_present": e.get("labels_present") or "",
                "source_path": e.get("source_path") or "",
                "bbox": e.get("bbox") or "",
            })

    return {"evidence": out}
