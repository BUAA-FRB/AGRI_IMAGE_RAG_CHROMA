# agri_rag/cli.py
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from .dataset import AgriVisionIndex
from .embedder import ClipEmbedder
from .image_utils import open_gray, open_rgb, nir_to_rgb
from .tiling import tile_image
from .chroma_store import get_or_create_collection, upsert, query
from .prompt import build_prompt
from .rag_predictor import (
    rag_predict_from_text,
    rag_predict_from_image,
    rag_retrieve_from_text,
    rag_retrieve_from_image,
)
from .qwen3_vl_client import Qwen3VLGenConfig
from .evaluator import EvalConfig, run_eval
from .reranker import QwenTextGenConfig
from .logging_utils import setup_logging

# NEW: allow cli to report last dumped json/montage paths
from .structured_io import get_last_dump_paths  # NEW

logger = logging.getLogger(__name__)


def md5_id(s: str) -> str:
    return hashlib.md5(s.encode("utf-8")).hexdigest()


def _infer_nir_from_rgb_path(rgb_path: str) -> Optional[str]:
    parts = rgb_path.replace("\\", "/")
    if "/field_images/rgb/" in parts:
        nir = parts.replace("/field_images/rgb/", "/field_images/nir/")
        nir = nir.replace("/", os.sep)
        if os.path.exists(nir):
            return os.path.abspath(nir)
    return None


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


def _build_rerank_gen(args: argparse.Namespace) -> Optional[QwenTextGenConfig]:
    if getattr(args, "rerank_mode", "none") != "qwen":
        return None

    max_new_tokens = int(getattr(args, "rerank_max_new_tokens", 1024))
    temperature = float(getattr(args, "rerank_temperature", 0.0))
    top_p = float(getattr(args, "rerank_top_p", 0.9))

    do_sample_str = getattr(args, "rerank_do_sample", "auto")
    if do_sample_str == "auto":
        do_sample = None
    elif do_sample_str == "true":
        do_sample = True
    else:
        do_sample = False

    return QwenTextGenConfig(
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        do_sample=do_sample,
    )


def _log_json(title: str, obj: Any, level: int = logging.INFO) -> None:
    try:
        s = json.dumps(obj, ensure_ascii=False, indent=2)
    except Exception:
        s = str(obj)
    logging.getLogger("agri_rag.output").log(level, "%s\n%s", title, s)


# =========================
# Indexing (RAG build)
# =========================
def cmd_index(args: argparse.Namespace) -> None:
    ds = AgriVisionIndex(args.dataset_root)
    embedder = ClipEmbedder(model_name=args.clip_model)

    col = get_or_create_collection(args.db_dir, args.collection)
    patch_col = None
    if args.patches:
        patch_col = get_or_create_collection(args.db_dir, args.patch_collection or "agri_patch")

    ids: List[str] = []
    embs: List[List[float]] = []
    metas: List[Dict[str, Any]] = []
    docs: List[str] = []

    patch_ids: List[str] = []
    patch_embs: List[List[float]] = []
    patch_metas: List[Dict[str, Any]] = []
    patch_docs: List[str] = []

    splits_filter = None
    if args.splits:
        splits_filter = [s.strip() for s in args.splits.split(",") if s.strip()]

    it = ds.iter_tiles(splits=splits_filter, max_items=args.max_items, fast_metadata=args.fast_metadata)
    recs = list(it)

    logger.info("Indexing start: dataset_root=%s items=%d patches=%s", args.dataset_root, len(recs), bool(args.patches))

    for idx, rec in enumerate(recs, 1):
        rgb = open_rgb(rec.rgb_path)
        nir = open_gray(rec.nir_path) if args.use_nir and rec.nir_path else None

        if nir is not None:
            emb = embedder.embed_image_rgb_nir(rgb, nir)
        else:
            emb = embedder.embed_image_rgb(rgb)

        rid = md5_id(f"{rec.extra.get('year')}|{rec.tile_id}|global")
        ids.append(rid)
        embs.append(emb)

        labels_present_str = json.dumps(rec.labels_present, ensure_ascii=False)
        label_areas_str = json.dumps(rec.label_areas, ensure_ascii=False)

        meta = {
            "type": "global",
            "tile_id": str(rec.tile_id),
            "split": str(rec.split) if rec.split is not None else "",
            "path": os.path.abspath(rec.rgb_path),
            "nir_path": os.path.abspath(rec.nir_path) if rec.nir_path else "",
            "year": str(rec.extra.get("year")) if rec.extra.get("year") is not None else "",
            "labels_present": labels_present_str,
            "label_areas": label_areas_str,
        }
        metas.append(meta)
        docs.append(f"tile_id={rec.tile_id} split={rec.split} labels={rec.labels_present}")

        if patch_col is not None:
            for i, (patch, bbox) in enumerate(tile_image(rgb, tile=args.tile, stride=args.stride)):
                if nir is not None:
                    nir_rgb = nir_to_rgb(nir)
                    nir_patch = nir_rgb.crop(bbox).convert("L")
                    emb_p = embedder.embed_image_rgb_nir(patch, nir_patch)
                else:
                    emb_p = embedder.embed_image_rgb(patch)

                pid = md5_id(f"{rec.extra.get('year')}|{rec.tile_id}|bbox={bbox}|i={i}")
                patch_ids.append(pid)
                patch_embs.append(emb_p)

                patch_metas.append({
                    "type": "patch",
                    "tile_id": str(rec.tile_id),
                    "split": str(rec.split) if rec.split is not None else "",
                    "path": os.path.abspath(rec.rgb_path),
                    "nir_path": os.path.abspath(rec.nir_path) if rec.nir_path else "",
                    "year": str(rec.extra.get("year")) if rec.extra.get("year") is not None else "",
                    "bbox": str(bbox),
                    "tile": int(args.tile),
                    "stride": int(args.stride),
                    "labels_present": labels_present_str,
                })
                patch_docs.append(f"tile_id={rec.tile_id} bbox={bbox} labels={rec.labels_present}")

        if idx % 50 == 0 or idx == len(recs):
            logger.info("Indexing progress: %d/%d", idx, len(recs))

    upsert(col, ids, embs, metas, docs)
    if patch_col is not None and patch_ids:
        upsert(patch_col, patch_ids, patch_embs, patch_metas, patch_docs)

    logger.info("Index done.")
    logger.info("- db_dir: %s", os.path.abspath(args.db_dir))
    logger.info("- collection: %s items=%d", args.collection, len(ids))
    if patch_col is not None:
        logger.info("- patch_collection: %s items=%d", args.patch_collection, len(patch_ids))


# =========================
# Retrieval (RAG query)
# =========================
def cmd_query_text(args: argparse.Namespace) -> None:
    hits, dbg = rag_retrieve_from_text(
        user_query=args.query,
        db_dir=args.db_dir,
        collection=args.collection,
        clip_model=args.clip_model,
        top_k=args.top_k,
        rerank_mode="none",
        stage1_k=args.stage1_k,
        enable_verify=args.enable_verify,
        verify_mode=args.verify_mode,
        use_patch_fusion=args.use_patch_fusion,
        patch_collection=args.patch_collection,
        fusion_mode=args.fusion_mode,
        fusion_rrf_k=args.fusion_rrf_k,
        global_top_k=args.global_top_k,
        patch_top_k=args.patch_top_k,
    )

    _log_json("query-text hits:", [h.__dict__ for h in hits], level=logging.INFO)
    if args.show_debug and dbg:
        _log_json("query-text debug:", dbg, level=logging.INFO)

    if args.build_prompt:
        prompt = build_prompt(args.query, hits)
        logging.getLogger("agri_rag.prompt").info("RAG prompt:\n%s", prompt)


def cmd_query_image(args: argparse.Namespace) -> None:
    hits, dbg = rag_retrieve_from_image(
        image_path=args.image_path,
        user_query="【图像查询】" + os.path.basename(args.image_path),
        db_dir=args.db_dir,
        collection=args.collection,
        clip_model=args.clip_model,
        top_k=args.top_k,
        use_nir_for_query=args.use_nir,
        nir_path=args.nir_path,
        exclude_self=False,
        rerank_mode="none",
        stage1_k=args.stage1_k,
        enable_verify=args.enable_verify,
        verify_mode=args.verify_mode,
        use_patch_fusion=args.use_patch_fusion,
        patch_collection=args.patch_collection,
        fusion_mode=args.fusion_mode,
        fusion_rrf_k=args.fusion_rrf_k,
        global_top_k=args.global_top_k,
        patch_top_k=args.patch_top_k,
    )

    _log_json("query-image hits:", [h.__dict__ for h in hits], level=logging.INFO)
    if args.show_debug and dbg:
        _log_json("query-image debug:", dbg, level=logging.INFO)

    if args.build_prompt:
        prompt = build_prompt("【图像查询】" + os.path.basename(args.image_path), hits)
        logging.getLogger("agri_rag.prompt").info("RAG prompt:\n%s", prompt)


# =========================
# Prompt-only (NO Qwen3-VL)
# =========================
def cmd_prompt_text(args: argparse.Namespace) -> None:
    hits, dbg = rag_retrieve_from_text(
        user_query=args.query,
        db_dir=args.db_dir,
        collection=args.collection,
        clip_model=args.clip_model,
        top_k=args.top_k,
        rerank_mode=args.rerank_mode,
        rerank_top_n=args.rerank_top_n,
        rerank_model=args.rerank_model,
        rerank_local_root=args.rerank_local_root,
        rerank_device_map=args.rerank_device_map,
        rerank_dtype=args.rerank_dtype,
        rerank_attn_impl=args.rerank_attn_impl,
        rerank_force_download=args.rerank_force_download,
        rerank_revision=args.rerank_revision,
        rerank_hf_token=args.rerank_hf_token,
        rerank_gen=_build_rerank_gen(args),
        stage1_k=args.stage1_k,
        enable_verify=args.enable_verify,
        verify_mode=args.verify_mode,
        use_patch_fusion=args.use_patch_fusion,
        patch_collection=args.patch_collection,
        fusion_mode=args.fusion_mode,
        fusion_rrf_k=args.fusion_rrf_k,
        global_top_k=args.global_top_k,
        patch_top_k=args.patch_top_k,
    )

    if args.show_hits:
        _log_json("prompt-text hits:", [h.__dict__ for h in hits], level=logging.INFO)

    prompt = build_prompt(args.query, hits, max_hits=args.evidence_k)
    logging.getLogger("agri_rag.prompt").info("RAG prompt:\n%s", prompt)

    if args.show_debug and dbg:
        _log_json("prompt-text debug:", dbg, level=logging.INFO)


def cmd_prompt_image(args: argparse.Namespace) -> None:
    hits, dbg = rag_retrieve_from_image(
        image_path=args.image_path,
        user_query=args.query,
        db_dir=args.db_dir,
        collection=args.collection,
        clip_model=args.clip_model,
        top_k=args.top_k,
        use_nir_for_query=args.use_nir,
        nir_path=args.nir_path,
        exclude_self=True,
        rerank_mode=args.rerank_mode,
        rerank_top_n=args.rerank_top_n,
        rerank_model=args.rerank_model,
        rerank_local_root=args.rerank_local_root,
        rerank_device_map=args.rerank_device_map,
        rerank_dtype=args.rerank_dtype,
        rerank_attn_impl=args.rerank_attn_impl,
        rerank_force_download=args.rerank_force_download,
        rerank_revision=args.rerank_revision,
        rerank_hf_token=args.rerank_hf_token,
        rerank_gen=_build_rerank_gen(args),
        stage1_k=args.stage1_k,
        enable_verify=args.enable_verify,
        verify_mode=args.verify_mode,
        use_patch_fusion=args.use_patch_fusion,
        patch_collection=args.patch_collection,
        fusion_mode=args.fusion_mode,
        fusion_rrf_k=args.fusion_rrf_k,
        global_top_k=args.global_top_k,
        patch_top_k=args.patch_top_k,
    )

    if args.show_hits:
        _log_json("prompt-image hits:", [h.__dict__ for h in hits], level=logging.INFO)

    title = "【图像查询】" + os.path.basename(args.image_path)
    prompt = build_prompt(title, hits, max_hits=args.evidence_k)
    logging.getLogger("agri_rag.prompt").info("RAG prompt:\n%s", prompt)

    if args.show_debug and dbg:
        _log_json("prompt-image debug:", dbg, level=logging.INFO)


# =========================
# Prediction (RAG -> Qwen3-VL)
# =========================
def cmd_predict_text(args: argparse.Namespace) -> None:
    qwen_model = _maybe_resolve_qwen_local_path(args.qwen_model, args.qwen_local_root)

    gen = Qwen3VLGenConfig(
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        do_sample=None,
    )

    out, hits, _messages, dbg = rag_predict_from_text(
        user_query=args.query,
        db_dir=args.db_dir,
        collection=args.collection,
        clip_model=args.clip_model,
        qwen_model=qwen_model,
        top_k=args.top_k,
        evidence_k=args.evidence_k,
        include_nir_evidence=args.include_nir_evidence,
        output_json=args.json,
        qwen_device_map=args.device_map,
        qwen_dtype=args.dtype,
        qwen_attn_impl=args.attn_impl,
        gen=gen,
        rerank_mode=args.rerank_mode,
        rerank_top_n=args.rerank_top_n,
        rerank_model=args.rerank_model,
        rerank_local_root=args.rerank_local_root,
        rerank_device_map=args.rerank_device_map,
        rerank_dtype=args.rerank_dtype,
        rerank_attn_impl=args.rerank_attn_impl,
        rerank_force_download=args.rerank_force_download,
        rerank_revision=args.rerank_revision,
        rerank_hf_token=args.rerank_hf_token,
        rerank_gen=_build_rerank_gen(args),
        stage1_k=args.stage1_k,
        enable_verify=args.enable_verify,
        verify_mode=args.verify_mode,
        use_patch_fusion=args.use_patch_fusion,
        patch_collection=args.patch_collection,
        fusion_mode=args.fusion_mode,
        fusion_rrf_k=args.fusion_rrf_k,
        global_top_k=args.global_top_k,
        patch_top_k=args.patch_top_k,
    )

    if args.show_hits:
        _log_json("predict-text hits:", [h.__dict__ for h in hits], level=logging.INFO)

    if args.show_prompt:
        prompt = build_prompt(args.query, hits, max_hits=args.evidence_k)
        logging.getLogger("agri_rag.prompt").info("RAG prompt:\n%s", prompt)

    if args.show_debug and dbg:
        _log_json("predict-text debug:", dbg, level=logging.INFO)

    # NEW: show output artifacts path
    try:
        paths = get_last_dump_paths() or {}
        if paths:
            logging.getLogger("agri_rag.result").info("predict-text artifacts:\n%s", json.dumps(paths, ensure_ascii=False, indent=2))
    except Exception:
        pass

    logging.getLogger("agri_rag.result").info("predict-text output:\n%s", out)


def cmd_predict_image(args: argparse.Namespace) -> None:
    qwen_model = _maybe_resolve_qwen_local_path(args.qwen_model, args.qwen_local_root)

    gen = Qwen3VLGenConfig(
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        do_sample=None,
    )

    out, hits, _messages, dbg = rag_predict_from_image(
        image_path=args.image_path,
        user_query=args.query,
        db_dir=args.db_dir,
        collection=args.collection,
        clip_model=args.clip_model,
        qwen_model=qwen_model,
        top_k=args.top_k,
        evidence_k=args.evidence_k,
        use_nir_for_query=args.use_nir,
        nir_path=args.nir_path,
        include_nir_evidence=args.include_nir_evidence,
        output_json=args.json,
        qwen_device_map=args.device_map,
        qwen_dtype=args.dtype,
        qwen_attn_impl=args.attn_impl,
        gen=gen,
        rerank_mode=args.rerank_mode,
        rerank_top_n=args.rerank_top_n,
        exclude_self=True,
        rerank_model=args.rerank_model,
        rerank_local_root=args.rerank_local_root,
        rerank_device_map=args.rerank_device_map,
        rerank_dtype=args.rerank_dtype,
        rerank_attn_impl=args.rerank_attn_impl,
        rerank_force_download=args.rerank_force_download,
        rerank_revision=args.rerank_revision,
        rerank_hf_token=args.rerank_hf_token,
        rerank_gen=_build_rerank_gen(args),
        stage1_k=args.stage1_k,
        enable_verify=args.enable_verify,
        verify_mode=args.verify_mode,
        use_patch_fusion=args.use_patch_fusion,
        patch_collection=args.patch_collection,
        fusion_mode=args.fusion_mode,
        fusion_rrf_k=args.fusion_rrf_k,
        global_top_k=args.global_top_k,
        patch_top_k=args.patch_top_k,
    )

    if args.show_hits:
        _log_json("predict-image hits:", [h.__dict__ for h in hits], level=logging.INFO)

    if args.show_prompt:
        prompt = build_prompt("【图像查询】" + os.path.basename(args.image_path), hits, max_hits=args.evidence_k)
        logging.getLogger("agri_rag.prompt").info("RAG prompt:\n%s", prompt)

    if args.show_debug and dbg:
        _log_json("predict-image debug:", dbg, level=logging.INFO)

    # NEW: show output artifacts path
    try:
        paths = get_last_dump_paths() or {}
        if paths:
            logging.getLogger("agri_rag.result").info("predict-image artifacts:\n%s", json.dumps(paths, ensure_ascii=False, indent=2))
    except Exception:
        pass

    logging.getLogger("agri_rag.result").info("predict-image output:\n%s", out)


# =========================
# Eval
# =========================
def cmd_eval(args: argparse.Namespace) -> None:
    ann = getattr(EvalConfig, "__annotations__", {}) or {}

    def put(kw: Dict[str, Any], name: str, value: Any) -> None:
        if name in ann:
            kw[name] = value

    kw: Dict[str, Any] = {}
    put(kw, "dataset_root", args.dataset_root)
    put(kw, "split", args.split)
    put(kw, "max_items", args.max_items)
    put(kw, "db_dir", args.db_dir)
    put(kw, "collection", args.collection)

    put(kw, "use_patch_fusion", args.use_patch_fusion)
    put(kw, "patch_collection", args.patch_collection)
    put(kw, "fusion_mode", args.fusion_mode)
    put(kw, "fusion_rrf_k", args.fusion_rrf_k)
    put(kw, "global_top_k", args.global_top_k)
    put(kw, "patch_top_k", args.patch_top_k)

    put(kw, "clip_model", args.clip_model)
    put(kw, "qwen_model", args.qwen_model)
    put(kw, "qwen_local_root", args.qwen_local_root)

    put(kw, "top_k", args.top_k)
    put(kw, "evidence_k", args.evidence_k)
    put(kw, "use_nir_for_query", args.use_nir)
    put(kw, "include_nir_evidence", args.include_nir_evidence)

    put(kw, "rerank_mode", args.rerank_mode)
    put(kw, "rerank_top_n", args.rerank_top_n)

    put(kw, "device_map", args.device_map)
    put(kw, "dtype", args.dtype)
    put(kw, "attn_impl", args.attn_impl)
    put(kw, "max_new_tokens", args.max_new_tokens)
    put(kw, "temperature", args.temperature)
    put(kw, "top_p", args.top_p)

    put(kw, "out_report", args.out_report)
    put(kw, "out_jsonl", args.out_jsonl)

    cfg = EvalConfig(**kw)
    report = run_eval(cfg)
    _log_json("eval report:", report, level=logging.INFO)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="agri_rag")
    sub = p.add_subparsers(dest="cmd", required=True)

    def add_logging_args(pp: argparse.ArgumentParser) -> None:
        pp.add_argument("--log_file", default="./logs/agri_rag.log", help="log file path (default ./logs/agri_rag.log)")
        pp.add_argument("--log_level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
        pp.add_argument("--log_to_console", action="store_true", help="also log to console (default off)")
        pp.add_argument("--log_max_mb", type=int, default=50)
        pp.add_argument("--log_backup_count", type=int, default=5)
        pp.add_argument("--no_redirect_std", action="store_true", help="do not redirect stdout/stderr to log")

    def add_fusion_args(pp: argparse.ArgumentParser) -> None:
        pp.add_argument("--use_patch_fusion", action="store_true", help="enable patch/global fusion retrieval")
        pp.add_argument("--patch_collection", default="agri_patch", help="patch collection name (default agri_patch)")
        pp.add_argument("--fusion_mode", default="rrf", choices=["rrf", "distance"], help="fusion strategy")
        pp.add_argument("--fusion_rrf_k", type=int, default=60, help="rrf parameter k (default 60)")

        pp.add_argument(
            "--stage1_k",
            type=int,
            default=None,
            help="override base stage-1 candidate size (None=auto; used as base before fusion/rerank)",
        )

        pp.add_argument(
            "--global_top_k",
            type=int,
            default=None,
            help="stage-1 candidates from global collection (None=auto, follow stage1_k/auto)",
        )
        pp.add_argument(
            "--patch_top_k",
            type=int,
            default=None,
            help="stage-1 candidates from patch collection (None=auto, follow stage1_k/auto)",
        )

    def add_verify_args(pp: argparse.ArgumentParser) -> None:
        pp.add_argument("--enable_verify", action="store_true", help="enable three-stage verification report in debug")
        pp.add_argument("--verify_mode", default="three_stage", choices=["three_stage"], help="verification mode")

    # index
    pi = sub.add_parser("index", help="Index Agriculture-Vision style dataset into Chroma")
    add_logging_args(pi)
    pi.add_argument("--dataset_root", required=True)
    pi.add_argument("--db_dir", default="./data/chroma_db")
    pi.add_argument("--collection", default="agri_global")
    pi.add_argument("--patch_collection", default="agri_patch")
    pi.add_argument("--patches", action="store_true")
    pi.add_argument("--tile", type=int, default=224)
    pi.add_argument("--stride", type=int, default=160)
    pi.add_argument("--use_nir", action="store_true")
    pi.add_argument("--fast_metadata", action="store_true")
    pi.add_argument("--max_items", type=int, default=None)
    pi.add_argument("--splits", default=None)
    pi.add_argument("--clip_model", default="openai/clip-vit-base-patch32")
    pi.set_defaults(func=cmd_index)

    # query text
    pqt = sub.add_parser("query-text", help="Text -> image retrieval (CLIP text encoder)")
    add_logging_args(pqt)
    add_fusion_args(pqt)
    add_verify_args(pqt)
    pqt.add_argument("--db_dir", default="./data/chroma_db")
    pqt.add_argument("--collection", default="agri_global")
    pqt.add_argument("--query", required=True)
    pqt.add_argument("--top_k", type=int, default=5)
    pqt.add_argument("--build_prompt", action="store_true")
    pqt.add_argument("--show_debug", action="store_true")
    pqt.add_argument("--clip_model", default="openai/clip-vit-base-patch32")
    pqt.set_defaults(func=cmd_query_text)

    # query image
    pqi = sub.add_parser("query-image", help="Image -> image retrieval (CLIP image encoder)")
    add_logging_args(pqi)
    add_fusion_args(pqi)
    add_verify_args(pqi)
    pqi.add_argument("--db_dir", default="./data/chroma_db")
    pqi.add_argument("--collection", default="agri_global")
    pqi.add_argument("--image_path", required=True)
    pqi.add_argument("--nir_path", default=None)
    pqi.add_argument("--use_nir", action="store_true")
    pqi.add_argument("--top_k", type=int, default=5)
    pqi.add_argument("--build_prompt", action="store_true")
    pqi.add_argument("--show_debug", action="store_true")
    pqi.add_argument("--clip_model", default="openai/clip-vit-base-patch32")
    pqi.set_defaults(func=cmd_query_image)

    def add_rerank_args(pp: argparse.ArgumentParser):
        pp.add_argument("--rerank_mode", default="none", choices=["none", "qwen"], help="two-stage retrieval rerank mode")
        pp.add_argument("--rerank_top_n", type=int, default=30, help="stage-1 candidates for rerank")
        pp.add_argument("--rerank_model", default="Qwen/Qwen2.5-1.5B-Instruct", help="HF repo id OR local dir")
        pp.add_argument("--rerank_local_root", default="./models", help="where to download reranker model snapshots")
        pp.add_argument("--rerank_device_map", default="auto")
        pp.add_argument("--rerank_dtype", default="auto")
        pp.add_argument("--rerank_attn_impl", default=None)
        pp.add_argument("--rerank_force_download", action="store_true")
        pp.add_argument("--rerank_revision", default=None)
        pp.add_argument("--rerank_hf_token", default=None)

        pp.add_argument("--rerank_max_new_tokens", type=int, default=1024)
        pp.add_argument("--rerank_temperature", type=float, default=0.0)
        pp.add_argument("--rerank_top_p", type=float, default=0.9)
        pp.add_argument("--rerank_do_sample", default="auto", choices=["auto", "true", "false"])

    # prompt-text
    ppt_only = sub.add_parser("prompt-text", help="Text -> retrieve (+optional rerank) -> log RAG prompt only")
    add_logging_args(ppt_only)
    add_fusion_args(ppt_only)
    add_verify_args(ppt_only)
    ppt_only.add_argument("--db_dir", default="./data/chroma_db")
    ppt_only.add_argument("--collection", default="agri_global")
    ppt_only.add_argument("--query", required=True)
    ppt_only.add_argument("--top_k", type=int, default=8)
    ppt_only.add_argument("--evidence_k", type=int, default=6)
    ppt_only.add_argument("--clip_model", default="openai/clip-vit-base-patch32")
    ppt_only.add_argument("--show_hits", action="store_true")
    ppt_only.add_argument("--show_debug", action="store_true")
    add_rerank_args(ppt_only)
    ppt_only.set_defaults(func=cmd_prompt_text)

    # prompt-image
    ppi_only = sub.add_parser("prompt-image", help="Image -> retrieve (+optional rerank) -> log RAG prompt only")
    add_logging_args(ppi_only)
    add_fusion_args(ppi_only)
    add_verify_args(ppi_only)
    ppi_only.add_argument("--db_dir", default="./data/chroma_db")
    ppi_only.add_argument("--collection", default="agri_global")
    ppi_only.add_argument("--image_path", required=True)
    ppi_only.add_argument("--query", default="Please identify the anomaly/disaster type and justify with retrieved evidence.")
    ppi_only.add_argument("--top_k", type=int, default=8)
    ppi_only.add_argument("--evidence_k", type=int, default=6)
    ppi_only.add_argument("--clip_model", default="openai/clip-vit-base-patch32")
    ppi_only.add_argument("--use_nir", action="store_true")
    ppi_only.add_argument("--nir_path", default=None)
    ppi_only.add_argument("--show_hits", action="store_true")
    ppi_only.add_argument("--show_debug", action="store_true")
    add_rerank_args(ppi_only)
    ppi_only.set_defaults(func=cmd_prompt_image)

    # predict-text
    ppt = sub.add_parser("predict-text", help="Text -> RAG retrieval -> Qwen3-VL disaster prediction")
    add_logging_args(ppt)
    add_fusion_args(ppt)
    add_verify_args(ppt)
    ppt.add_argument("--db_dir", default="./data/chroma_db")
    ppt.add_argument("--collection", default="agri_global")
    ppt.add_argument("--query", required=True)
    ppt.add_argument("--top_k", type=int, default=5)
    ppt.add_argument("--evidence_k", type=int, default=6)
    ppt.add_argument("--clip_model", default="openai/clip-vit-base-patch32")

    ppt.add_argument("--qwen_model", default="Qwen/Qwen3-VL-4B-Instruct")
    ppt.add_argument("--qwen_local_root", default="./models")

    ppt.add_argument("--device_map", default="auto")
    ppt.add_argument("--dtype", default="auto")
    ppt.add_argument("--attn_impl", default=None)
    ppt.add_argument("--max_new_tokens", type=int, default=512)
    ppt.add_argument("--temperature", type=float, default=0.2)
    ppt.add_argument("--top_p", type=float, default=0.9)

    ppt.add_argument("--include_nir_evidence", action="store_true")
    ppt.add_argument("--json", action="store_true")
    ppt.add_argument("--show_hits", action="store_true")
    ppt.add_argument("--show_prompt", action="store_true")
    ppt.add_argument("--show_debug", action="store_true")
    add_rerank_args(ppt)
    ppt.set_defaults(func=cmd_predict_text)

    # predict-image
    ppi = sub.add_parser("predict-image", help="Image -> RAG retrieval -> Qwen3-VL disaster prediction")
    add_logging_args(ppi)
    add_fusion_args(ppi)
    add_verify_args(ppi)
    ppi.add_argument("--db_dir", default="./data/chroma_db")
    ppi.add_argument("--collection", default="agri_global")
    ppi.add_argument("--image_path", required=True)
    ppi.add_argument("--query", default="请识别该农田图像中的灾害/异常类型，并结合检索证据给出理由。")
    ppi.add_argument("--top_k", type=int, default=5)
    ppi.add_argument("--evidence_k", type=int, default=6)
    ppi.add_argument("--clip_model", default="openai/clip-vit-base-patch32")

    ppi.add_argument("--use_nir", action="store_true")
    ppi.add_argument("--nir_path", default=None)

    ppi.add_argument("--qwen_model", default="Qwen/Qwen3-VL-4B-Instruct")
    ppi.add_argument("--qwen_local_root", default="./models")

    ppi.add_argument("--device_map", default="auto")
    ppi.add_argument("--dtype", default="auto")
    ppi.add_argument("--attn_impl", default=None)
    ppi.add_argument("--max_new_tokens", type=int, default=512)
    ppi.add_argument("--temperature", type=float, default=0.2)
    ppi.add_argument("--top_p", type=float, default=0.9)

    ppi.add_argument("--include_nir_evidence", action="store_true")
    ppi.add_argument("--json", action="store_true")
    ppi.add_argument("--show_hits", action="store_true")
    ppi.add_argument("--show_prompt", action="store_true")
    ppi.add_argument("--show_debug", action="store_true")
    add_rerank_args(ppi)
    ppi.set_defaults(func=cmd_predict_image)

    # eval
    pe = sub.add_parser("eval", help="Evaluate RAG->Qwen prediction on a dataset split")
    add_logging_args(pe)
    add_fusion_args(pe)
    pe.add_argument("--dataset_root", required=True)
    pe.add_argument("--split", default="test", choices=["train", "val", "test"])
    pe.add_argument("--max_items", type=int, default=None)

    pe.add_argument("--db_dir", default="./data/chroma_db")
    pe.add_argument("--collection", default="agri_global")
    pe.add_argument("--clip_model", default="openai/clip-vit-base-patch32")

    pe.add_argument("--qwen_model", default="Qwen/Qwen3-VL-4B-Instruct")
    pe.add_argument("--qwen_local_root", default="./models")

    pe.add_argument("--top_k", type=int, default=8)
    pe.add_argument("--evidence_k", type=int, default=6)

    pe.add_argument("--use_nir", action="store_true")
    pe.add_argument("--include_nir_evidence", action="store_true")

    pe.add_argument("--device_map", default="auto")
    pe.add_argument("--dtype", default="auto")
    pe.add_argument("--attn_impl", default=None)
    pe.add_argument("--max_new_tokens", type=int, default=256)
    pe.add_argument("--temperature", type=float, default=0.0)
    pe.add_argument("--top_p", type=float, default=0.9)

    pe.add_argument("--rerank_mode", default="none", choices=["none", "qwen"])
    pe.add_argument("--rerank_top_n", type=int, default=30)

    pe.add_argument("--out_report", default="./eval_report.json")
    pe.add_argument("--out_jsonl", default="./eval_cases.jsonl")
    pe.set_defaults(func=cmd_eval)

    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    setup_logging(
        log_file=getattr(args, "log_file", "./logs/agri_rag.log"),
        level=getattr(args, "log_level", "INFO"),
        to_console=bool(getattr(args, "log_to_console", False)),
        max_mb=int(getattr(args, "log_max_mb", 50)),
        backup_count=int(getattr(args, "log_backup_count", 5)),
        redirect_std=not bool(getattr(args, "no_redirect_std", False)),
    )

    run_logger = logging.getLogger("agri_rag.run")
    run_logger.info("Command start: %s", vars(args))

    try:
        args.func(args)
        run_logger.info("Command done: %s", args.cmd)
    except SystemExit:
        raise
    except Exception:
        run_logger.exception("Command failed: %s", args.cmd)
        sys.exit(1)


if __name__ == "__main__":
    main()
