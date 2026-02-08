from typing import Any, Dict, List, Optional

from .output_reader import extract_prediction_fields, extract_evidence_hint
from .kb_retriever import Retriever
from .prompts import build_messages
from .structured_io import extract_json_object
from .safety import postprocess_advice_json
from .text_llm_client import Qwen25TextClient, GenConfig
from .label_normalizer import LabelNormalizer


def build_retrieval_query(pred: Dict[str, Any], normalizer: LabelNormalizer) -> Dict[str, Any]:
    raw_label = str(pred.get("pred_label") or "")
    raw_labels = pred.get("labels") or []
    raw_labels = [raw_label] + [str(x) for x in raw_labels if str(x).strip()]

    canon_labels = normalizer.normalize_many(raw_labels)
    canon_label = canon_labels[0] if canon_labels else normalizer.normalize_one(raw_label)

    expanded_tags, expanded_kws = normalizer.expand_for_retrieval([canon_label])

    base_terms = [canon_label] + expanded_tags + expanded_kws + [
        "灾前", "预防", "灾中", "应急处置", "灾后", "损失统计", "恢复重建", "监测复盘",
        "田间踏查", "取证", "面积估算", "等级评估",
        "田间管理", "排水", "灌溉", "补播", "复种", "病虫草害", "农艺措施",
    ]
    extra = (pred.get("conclusion") or "") + " " + (pred.get("uncertainty") or "")
    query = " ".join([t for t in base_terms if str(t).strip()]) + " " + extra.strip()

    return {
        "canon_label": canon_label,
        "canon_labels": canon_labels,
        "query": query.strip(),
        "query_tags": expanded_tags,
    }


def run_expert_advice(
    final_output: Dict[str, Any],
    db_dir: str,
    collection: str,
    embed_model: str,
    qwen25_model_path: Optional[str] = None,
    kb_top_k: int = 6,
    enable_llm: bool = True,
    label_map_path: Optional[str] = None,
) -> Dict[str, Any]:
    pred = extract_prediction_fields(final_output)
    evidence_hint = extract_evidence_hint(final_output)

    normalizer = LabelNormalizer(label_map_path=label_map_path)
    q = build_retrieval_query(pred, normalizer)

    retriever = Retriever(db_dir=db_dir, collection=collection, embed_model=embed_model)
    retrieved = retriever.search(q["query"], top_k=kb_top_k, query_tags=q["query_tags"])

    result: Dict[str, Any] = {
        "schema": "advice.run_output.v3_chroma_norm",
        "pred": {
            **pred,
            "canon_label": q["canon_label"],
            "canon_labels": q["canon_labels"],
        },
        "normalization": {
            "label_map_path": label_map_path or "",
            "query_tags_used": q["query_tags"],
        },
        "retrieval": {
            "query": q["query"],
            "top_k": kb_top_k,
            "hits": [
                {
                    "rank": i + 1,
                    "kid": h.get("kid"),
                    "score": h.get("score"),
                    "distance": h.get("distance"),
                    "meta": h.get("meta"),
                    "text": h.get("text"),
                }
                for i, h in enumerate(retrieved)
            ],
        },
        "generation": {
            "enabled": bool(enable_llm),
            "raw_text": "",
            "parse_ok": False,
            "parsed_json": None,
        },
    }

    if not enable_llm:
        return result

    if not qwen25_model_path:
        raise RuntimeError("enable_llm=True 时必须提供 qwen25_model_path（本地 Qwen2.5-3B 目录）")

    client = Qwen25TextClient(model_path=qwen25_model_path)
    messages = build_messages(result["pred"], evidence_hint, retrieved)
    raw = client.chat(messages, gen=GenConfig())
    parsed = extract_json_object(raw)

    result["generation"]["raw_text"] = raw
    result["generation"]["parse_ok"] = parsed is not None
    if parsed is not None:
        result["generation"]["parsed_json"] = postprocess_advice_json(parsed)

    return result
