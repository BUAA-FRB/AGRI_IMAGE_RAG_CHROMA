# agri_rag/evaluator.py
from __future__ import annotations

import json
import os
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .dataset import AgriVisionIndex
from .structured_io import extract_json_object, normalize_predicted_labels, dedup_keep_order
from .rag_predictor import rag_predict_from_image
from .qwen3_vl_client import Qwen3VLGenConfig

logger = logging.getLogger(__name__)


def _maybe_resolve_qwen_local_path(qwen_model: str, qwen_local_root: str) -> str:
    try:
        p = Path(qwen_model)
        if p.exists() and p.is_dir():
            return str(p.resolve())
    except Exception:
        pass

    if "/" in qwen_model or "\\" in qwen_model:
        local_dir = Path(qwen_local_root) / Path(qwen_model.replace("/", os.sep).replace("\\", os.sep))
        if local_dir.exists() and local_dir.is_dir():
            return str(local_dir.resolve())

    return qwen_model


def _set_metrics(pred: List[str], gt: List[str]) -> Tuple[int, int, int]:
    ps = set(pred)
    gs = set(gt)
    tp = len(ps & gs)
    fp = len(ps - gs)
    fn = len(gs - ps)
    return tp, fp, fn


def _parse_labels_present(labels_present: Any) -> List[str]:
    """
    labels_present 在库里通常是 JSON 字符串：["endrow","double_plant"]
    这里尽量解析为 label 列表；失败则做简单拆分兜底。
    """
    if labels_present is None:
        return []
    if isinstance(labels_present, list):
        out = []
        for it in labels_present:
            if isinstance(it, str) and it.strip():
                out.append(it.strip())
        return dedup_keep_order(out)

    if not isinstance(labels_present, str):
        return []

    s = labels_present.strip()
    if not s:
        return []
    try:
        obj = json.loads(s)
        if isinstance(obj, list):
            out = []
            for it in obj:
                if isinstance(it, str) and it.strip():
                    out.append(it.strip())
            return dedup_keep_order(out)
    except Exception:
        pass

    for sep in [",", "，", "、", "|", ";", "/"]:
        if sep in s:
            parts = [p.strip().strip('"').strip("'") for p in s.split(sep)]
            parts = [p for p in parts if p]
            return dedup_keep_order(parts)
    return [s]


def _union_labels_from_hit_briefs(hit_briefs: List[Dict[str, Any]]) -> List[str]:
    labs: List[str] = []
    for hb in hit_briefs or []:
        lbs = _parse_labels_present(hb.get("labels_present", ""))
        labs.extend(lbs)
    return dedup_keep_order([x for x in labs if x])


def _agg_update_per_label(per_label: Dict[str, Dict[str, int]], pred: List[str], gt: List[str]) -> None:
    ps = set(pred)
    gs = set(gt)
    for lb in ps | gs:
        if lb not in per_label:
            per_label[lb] = {"tp": 0, "fp": 0, "fn": 0}
        if lb in ps and lb in gs:
            per_label[lb]["tp"] += 1
        elif lb in ps and lb not in gs:
            per_label[lb]["fp"] += 1
        elif lb not in ps and lb in gs:
            per_label[lb]["fn"] += 1


def _finalize_stage(sum_tp: int, sum_fp: int, sum_fn: int, exact_match: int, total: int) -> Dict[str, Any]:
    prec = (sum_tp / (sum_tp + sum_fp)) if (sum_tp + sum_fp) > 0 else 0.0
    rec_ = (sum_tp / (sum_tp + sum_fn)) if (sum_tp + sum_fn) > 0 else 0.0
    f1 = (2 * prec * rec_ / (prec + rec_)) if (prec + rec_) > 0 else 0.0
    subset_acc = (exact_match / total) if total > 0 else 0.0
    return {
        "micro_precision": prec,
        "micro_recall": rec_,
        "micro_f1": f1,
        "subset_accuracy": subset_acc,
        "tp": sum_tp,
        "fp": sum_fp,
        "fn": sum_fn,
        "exact_match": exact_match,
    }


@dataclass
class EvalConfig:
    dataset_root: str
    split: str = "test"
    max_items: Optional[int] = None

    db_dir: str = "./data/chroma_db"
    collection: str = "agri_global"

    # patch/global fusion
    use_patch_fusion: bool = False
    patch_collection: str = "agri_patch"
    fusion_mode: str = "rrf"
    fusion_rrf_k: int = 60

    clip_model: str = "openai/clip-vit-base-patch32"
    qwen_model: str = "Qwen/Qwen3-VL-4B-Instruct"
    qwen_local_root: str = "./models"

    top_k: int = 8
    evidence_k: int = 6

    use_nir_for_query: bool = False
    include_nir_evidence: bool = False

    rerank_mode: str = "none"
    rerank_top_n: int = 30

    device_map: str = "auto"
    dtype: str = "auto"
    attn_impl: Optional[str] = None

    max_new_tokens: int = 256
    temperature: float = 0.0
    top_p: float = 0.9

    out_report: str = "./eval_report.json"
    out_jsonl: Optional[str] = "./eval_cases.jsonl"


def run_eval(cfg: EvalConfig) -> Dict[str, Any]:
    logger.info("Eval start: %s", cfg.__dict__)

    ds = AgriVisionIndex(cfg.dataset_root)
    qwen_model = _maybe_resolve_qwen_local_path(cfg.qwen_model, cfg.qwen_local_root)

    gen = Qwen3VLGenConfig(
        max_new_tokens=cfg.max_new_tokens,
        temperature=cfg.temperature,
        top_p=cfg.top_p,
        do_sample=None,
    )

    it = ds.iter_tiles(splits=[cfg.split], max_items=cfg.max_items, fast_metadata=False)

    total = 0

    # 三段：retrieval / rerank / generation
    s1_tp = s1_fp = s1_fn = 0
    s2_tp = s2_fp = s2_fn = 0
    s3_tp = s3_fp = s3_fn = 0
    s1_exact = s2_exact = s3_exact = 0

    per_label_s1: Dict[str, Dict[str, int]] = {}
    per_label_s2: Dict[str, Dict[str, int]] = {}
    per_label_s3: Dict[str, Dict[str, int]] = {}

    all_labels: List[str] = ds.label_names[:] if ds.label_names else []

    cases: List[Dict[str, Any]] = []

    for rec in it:
        total += 1
        gt = dedup_keep_order([str(x) for x in (rec.labels_present or [])])
        for lb in gt:
            if lb not in all_labels:
                all_labels.append(lb)

        user_query = "请识别该农田图像中的灾害/异常类型，并结合检索证据给出理由。"

        raw_out, hits, _messages, dbg = rag_predict_from_image(
            image_path=rec.rgb_path,
            user_query=user_query,
            db_dir=cfg.db_dir,
            collection=cfg.collection,
            clip_model=cfg.clip_model,
            qwen_model=qwen_model,
            top_k=cfg.top_k,
            evidence_k=cfg.evidence_k,
            use_nir_for_query=cfg.use_nir_for_query,
            nir_path=rec.nir_path if cfg.use_nir_for_query else None,
            include_nir_evidence=cfg.include_nir_evidence,
            output_json=True,
            qwen_device_map=cfg.device_map,
            qwen_dtype=cfg.dtype,
            qwen_attn_impl=cfg.attn_impl,
            gen=gen,
            rerank_mode=cfg.rerank_mode,
            rerank_top_n=cfg.rerank_top_n,
            exclude_self=True,
            # fusion
            use_patch_fusion=cfg.use_patch_fusion,
            patch_collection=cfg.patch_collection,
            fusion_mode=cfg.fusion_mode,
            fusion_rrf_k=cfg.fusion_rrf_k,
        )

        # -------- Stage-1 (检索：rerank 前 top_k) --------
        s1_briefs = (dbg.get("stage1", {}) or {}).get("before_topk", []) or []
        s1_pred = _union_labels_from_hit_briefs(s1_briefs)
        tp, fp, fn = _set_metrics(s1_pred, gt)
        s1_tp += tp
        s1_fp += fp
        s1_fn += fn
        if set(s1_pred) == set(gt):
            s1_exact += 1
        _agg_update_per_label(per_label_s1, s1_pred, gt)

        # -------- Stage-2 (重排：rerank 后 top_k；若没开 rerank，则等同 final) --------
        if "rerank" in dbg and isinstance(dbg.get("rerank", None), dict) and dbg["rerank"].get("after_topk", None) is not None:
            s2_briefs = dbg["rerank"].get("after_topk", []) or []
        else:
            s2_briefs = (dbg.get("final", {}) or {}).get("after_topk", []) or s1_briefs
        s2_pred = _union_labels_from_hit_briefs(s2_briefs)
        tp, fp, fn = _set_metrics(s2_pred, gt)
        s2_tp += tp
        s2_fp += fp
        s2_fn += fn
        if set(s2_pred) == set(gt):
            s2_exact += 1
        _agg_update_per_label(per_label_s2, s2_pred, gt)

        # -------- Stage-3 (生成：Qwen 输出 labels) --------
        obj = extract_json_object(raw_out) or {}
        pred = normalize_predicted_labels(obj)
        tp, fp, fn = _set_metrics(pred, gt)
        s3_tp += tp
        s3_fp += fp
        s3_fn += fn
        if set(pred) == set(gt):
            s3_exact += 1
        _agg_update_per_label(per_label_s3, pred, gt)

        cases.append({
            "tile_id": rec.tile_id,
            "split": rec.split,
            "rgb_path": os.path.abspath(rec.rgb_path),
            "gt_labels": gt,
            "stage1_retrieval_labels": s1_pred,
            "stage2_rerank_labels": s2_pred,
            "stage3_generation_labels": pred,
            "qwen_raw": raw_out,
            "qwen_json": obj,
            "hits": [h.__dict__ for h in hits],
            "debug": dbg,
        })

        if total % 20 == 0:
            logger.info("Eval progress: %d cases processed", total)

    report = {
        "split": cfg.split,
        "total": total,
        "stages": {
            "retrieval": _finalize_stage(s1_tp, s1_fp, s1_fn, s1_exact, total),
            "rerank": _finalize_stage(s2_tp, s2_fp, s2_fn, s2_exact, total),
            "generation": _finalize_stage(s3_tp, s3_fp, s3_fn, s3_exact, total),
        },
        "per_label": {
            "retrieval": per_label_s1,
            "rerank": per_label_s2,
            "generation": per_label_s3,
        },
        "config": cfg.__dict__,
    }

    Path(cfg.out_report).parent.mkdir(parents=True, exist_ok=True)
    with open(cfg.out_report, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    if cfg.out_jsonl:
        Path(cfg.out_jsonl).parent.mkdir(parents=True, exist_ok=True)
        with open(cfg.out_jsonl, "w", encoding="utf-8") as f:
            for c in cases:
                f.write(json.dumps(c, ensure_ascii=False) + "\n")

    logger.info(
        "Eval done: total=%d | retrieval_f1=%.4f rerank_f1=%.4f gen_f1=%.4f | out_report=%s out_jsonl=%s",
        total,
        report["stages"]["retrieval"]["micro_f1"],
        report["stages"]["rerank"]["micro_f1"],
        report["stages"]["generation"]["micro_f1"],
        cfg.out_report,
        cfg.out_jsonl,
    )

    return report
