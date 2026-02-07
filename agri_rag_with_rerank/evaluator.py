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


@dataclass
class EvalConfig:
    dataset_root: str
    split: str = "test"
    max_items: Optional[int] = None

    db_dir: str = "./data/chroma_db"
    collection: str = "agri_global"

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
    sum_tp = sum_fp = sum_fn = 0
    exact_match = 0

    per_label: Dict[str, Dict[str, int]] = {}
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
        )

        obj = extract_json_object(raw_out) or {}
        pred = normalize_predicted_labels(obj)

        tp, fp, fn = _set_metrics(pred, gt)
        sum_tp += tp
        sum_fp += fp
        sum_fn += fn

        if set(pred) == set(gt):
            exact_match += 1

        for lb in set(pred) | set(gt):
            if lb not in per_label:
                per_label[lb] = {"tp": 0, "fp": 0, "fn": 0}
            if lb in pred and lb in gt:
                per_label[lb]["tp"] += 1
            elif lb in pred and lb not in gt:
                per_label[lb]["fp"] += 1
            elif lb not in pred and lb in gt:
                per_label[lb]["fn"] += 1

        cases.append({
            "tile_id": rec.tile_id,
            "split": rec.split,
            "rgb_path": os.path.abspath(rec.rgb_path),
            "gt_labels": gt,
            "pred_labels": pred,
            "qwen_raw": raw_out,
            "qwen_json": obj,
            "hits": [h.__dict__ for h in hits],
            "debug": dbg,
        })

        if total % 20 == 0:
            logger.info("Eval progress: %d cases processed", total)

    prec = (sum_tp / (sum_tp + sum_fp)) if (sum_tp + sum_fp) > 0 else 0.0
    rec_ = (sum_tp / (sum_tp + sum_fn)) if (sum_tp + sum_fn) > 0 else 0.0
    f1 = (2 * prec * rec_ / (prec + rec_)) if (prec + rec_) > 0 else 0.0
    subset_acc = (exact_match / total) if total > 0 else 0.0

    report = {
        "split": cfg.split,
        "total": total,
        "micro_precision": prec,
        "micro_recall": rec_,
        "micro_f1": f1,
        "subset_accuracy": subset_acc,
        "per_label": per_label,
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

    logger.info("Eval done: total=%d micro_f1=%.4f subset_acc=%.4f out_report=%s out_jsonl=%s",
                total, f1, subset_acc, cfg.out_report, cfg.out_jsonl)

    return report
