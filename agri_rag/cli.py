from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from tqdm import tqdm

from .dataset import AgriVisionIndex
from .embedder import ClipEmbedder
from .image_utils import open_gray, open_rgb, nir_to_rgb
from .tiling import tile_image
from .chroma_store import get_or_create_collection, upsert, query
from .prompt import build_prompt
from .rag_predictor import rag_predict_from_text, rag_predict_from_image
from .qwen3_vl_client import Qwen3VLGenConfig


def md5_id(s: str) -> str:
    return hashlib.md5(s.encode("utf-8")).hexdigest()


def _infer_nir_from_rgb_path(rgb_path: str) -> Optional[str]:
    # Heuristic: replace /rgb/ with /nir/
    parts = rgb_path.replace("\\", "/")
    if "/field_images/rgb/" in parts:
        nir = parts.replace("/field_images/rgb/", "/field_images/nir/")
        nir = nir.replace("/", os.sep)
        if os.path.exists(nir):
            return os.path.abspath(nir)
    return None


def _maybe_resolve_qwen_local_path(qwen_model: str, qwen_local_root: str) -> str:
    """
    If user passes a HF repo id like "Qwen/Qwen3-VL-4B-Instruct",
    and local_root has "./models/Qwen/Qwen3-VL-4B-Instruct", then use local path.
    If user passes an existing local directory already, use it directly.
    Otherwise, keep as-is (might trigger download in Qwen3VLClient depending on its logic).
    """
    # Already a local dir?
    try:
        p = Path(qwen_model)
        if p.exists() and p.is_dir():
            return str(p.resolve())
    except Exception:
        pass

    # Repo id -> local root mirror
    if "/" in qwen_model or "\\" in qwen_model:
        # normalize "Qwen/Qwen3-VL-4B-Instruct" -> ./models/Qwen/Qwen3-VL-4B-Instruct
        local_dir = Path(qwen_local_root) / Path(qwen_model.replace("/", os.sep).replace("\\", os.sep))
        if local_dir.exists() and local_dir.is_dir():
            return str(local_dir.resolve())

    return qwen_model


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

    for rec in tqdm(list(it), desc="Indexing"):
        rgb = open_rgb(rec.rgb_path)
        nir = open_gray(rec.nir_path) if args.use_nir and rec.nir_path else None

        if nir is not None:
            emb = embedder.embed_image_rgb_nir(rgb, nir)
        else:
            emb = embedder.embed_image_rgb(rgb)

        rid = md5_id(f"{rec.extra.get('year')}|{rec.tile_id}|global")
        ids.append(rid)
        embs.append(emb)

        # ---- IMPORTANT: Chroma metadata values must be str/int/float/bool (no list/dict/None) ----
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
                    # also tile NIR and fuse
                    nir_rgb = nir_to_rgb(nir)
                    nir_patch = nir_rgb.crop(bbox).convert("L")  # back to gray for API
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
                    "bbox": str(bbox),          # tuple -> str (safe for Chroma metadata)
                    "tile": int(args.tile),
                    "stride": int(args.stride),
                    "labels_present": labels_present_str,  # list -> json str
                })
                patch_docs.append(f"tile_id={rec.tile_id} bbox={bbox} labels={rec.labels_present}")

    # write to chroma
    upsert(col, ids, embs, metas, docs)
    if patch_col is not None and patch_ids:
        upsert(patch_col, patch_ids, patch_embs, patch_metas, patch_docs)

    print("\nIndex done.")
    print(f"- db_dir: {os.path.abspath(args.db_dir)}")
    print(f"- collection: {args.collection}  items={len(ids)}")
    if patch_col is not None:
        print(f"- patch_collection: {args.patch_collection}  items={len(patch_ids)}")


# =========================
# Retrieval (RAG query)
# =========================
def cmd_query_text(args: argparse.Namespace) -> None:
    embedder = ClipEmbedder(model_name=args.clip_model)
    col = get_or_create_collection(args.db_dir, args.collection)

    qemb = embedder.embed_text(args.query)
    hits = query(col, qemb, top_k=args.top_k)

    out = [h.__dict__ for h in hits]
    print(json.dumps(out, ensure_ascii=False, indent=2))

    if args.build_prompt:
        print("\n" + "=" * 80)
        print(build_prompt(args.query, hits))
        print("=" * 80)


def cmd_query_image(args: argparse.Namespace) -> None:
    embedder = ClipEmbedder(model_name=args.clip_model)
    col = get_or_create_collection(args.db_dir, args.collection)

    rgb = open_rgb(args.image_path)
    nir_path = args.nir_path
    if args.use_nir and not nir_path:
        nir_path = _infer_nir_from_rgb_path(args.image_path)
    nir = open_gray(nir_path) if args.use_nir and nir_path and os.path.exists(nir_path) else None

    if nir is not None:
        qemb = embedder.embed_image_rgb_nir(rgb, nir)
    else:
        qemb = embedder.embed_image_rgb(rgb)

    hits = query(col, qemb, top_k=args.top_k)
    out = [h.__dict__ for h in hits]
    print(json.dumps(out, ensure_ascii=False, indent=2))

    if args.build_prompt:
        print("\n" + "=" * 80)
        print(build_prompt("【图像查询】" + os.path.basename(args.image_path), hits))
        print("=" * 80)


# =========================
# Prediction (RAG -> Qwen3-VL)
# =========================
def cmd_predict_text(args: argparse.Namespace) -> None:
    # Prefer local path if already downloaded under qwen_local_root
    qwen_model = _maybe_resolve_qwen_local_path(args.qwen_model, args.qwen_local_root)

    gen = Qwen3VLGenConfig(
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        do_sample=None,
    )

    out, hits, _messages = rag_predict_from_text(
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
    )

    if args.show_hits:
        print(json.dumps([h.__dict__ for h in hits], ensure_ascii=False, indent=2))

    if args.show_prompt:
        print("\n" + "=" * 80)
        print(build_prompt(args.query, hits, max_hits=args.evidence_k))
        print("=" * 80 + "\n")

    print(out)


def cmd_predict_image(args: argparse.Namespace) -> None:
    qwen_model = _maybe_resolve_qwen_local_path(args.qwen_model, args.qwen_local_root)

    gen = Qwen3VLGenConfig(
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        do_sample=None,
    )

    out, hits, _messages = rag_predict_from_image(
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
    )

    if args.show_hits:
        print(json.dumps([h.__dict__ for h in hits], ensure_ascii=False, indent=2))

    if args.show_prompt:
        print("\n" + "=" * 80)
        print(build_prompt("【图像查询】" + os.path.basename(args.image_path), hits, max_hits=args.evidence_k))
        print("=" * 80 + "\n")

    print(out)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="agri_rag")
    sub = p.add_subparsers(dest="cmd", required=True)

    # index
    pi = sub.add_parser("index", help="Index Agriculture-Vision style dataset into Chroma")
    pi.add_argument("--dataset_root", required=True, help="dataset root path (e.g., data2019_miniscale/)")
    pi.add_argument("--db_dir", default="./data/chroma_db", help="Chroma persistent directory")
    pi.add_argument("--collection", default="agri_global", help="collection name for global embeddings")
    pi.add_argument("--patch_collection", default="agri_patch", help="collection name for patch embeddings")
    pi.add_argument("--patches", action="store_true", help="also build patch-level collection")
    pi.add_argument("--tile", type=int, default=224, help="patch tile size")
    pi.add_argument("--stride", type=int, default=160, help="patch stride")
    pi.add_argument("--use_nir", action="store_true", help="fuse RGB and NIR embeddings (avg + renorm)")
    pi.add_argument("--fast_metadata", action="store_true", help="skip reading label masks (faster)")
    pi.add_argument("--max_items", type=int, default=None, help="index only first N tiles (debug)")
    pi.add_argument("--splits", default=None, help="comma separated: train,val,test (default all)")
    pi.add_argument("--clip_model", default="openai/clip-vit-base-patch32")
    pi.set_defaults(func=cmd_index)

    # query text
    pqt = sub.add_parser("query-text", help="Text -> image retrieval (CLIP text encoder)")
    pqt.add_argument("--db_dir", default="./data/chroma_db")
    pqt.add_argument("--collection", default="agri_global")
    pqt.add_argument("--query", required=True)
    pqt.add_argument("--top_k", type=int, default=5)
    pqt.add_argument("--build_prompt", action="store_true", help="also output a RAG prompt for LLMs")
    pqt.add_argument("--clip_model", default="openai/clip-vit-base-patch32")
    pqt.set_defaults(func=cmd_query_text)

    # query image
    pqi = sub.add_parser("query-image", help="Image -> image retrieval (CLIP image encoder)")
    pqi.add_argument("--db_dir", default="./data/chroma_db")
    pqi.add_argument("--collection", default="agri_global")
    pqi.add_argument("--image_path", required=True)
    pqi.add_argument("--nir_path", default=None, help="optional explicit NIR path")
    pqi.add_argument("--use_nir", action="store_true", help="fuse RGB+NIR for the query")
    pqi.add_argument("--top_k", type=int, default=5)
    pqi.add_argument("--build_prompt", action="store_true")
    pqi.add_argument("--clip_model", default="openai/clip-vit-base-patch32")
    pqi.set_defaults(func=cmd_query_image)

    # predict text (RAG -> Qwen3-VL)
    ppt = sub.add_parser("predict-text", help="Text -> RAG retrieval -> Qwen3-VL disaster prediction")
    ppt.add_argument("--db_dir", default="./data/chroma_db")
    ppt.add_argument("--collection", default="agri_global")
    ppt.add_argument("--query", required=True, help="user question / description")
    ppt.add_argument("--top_k", type=int, default=5, help="retrieval top_k")
    ppt.add_argument("--evidence_k", type=int, default=6, help="how many retrieved items to attach as evidence images")
    ppt.add_argument("--clip_model", default="openai/clip-vit-base-patch32")

    # Qwen model (repo id OR local path)
    ppt.add_argument("--qwen_model", default="Qwen/Qwen3-VL-4B-Instruct",
                     help="HF repo id (e.g., Qwen/Qwen3-VL-4B-Instruct) OR local directory path")
    ppt.add_argument("--qwen_local_root", default="./models",
                     help="if qwen_model is repo id, try local path under this root first (default ./models)")

    # Qwen runtime configs
    ppt.add_argument("--device_map", default="auto")
    ppt.add_argument("--dtype", default="auto", help="auto/float16/bfloat16...")
    ppt.add_argument("--attn_impl", default=None, help="e.g. flash_attention_2 / sdpa")
    ppt.add_argument("--max_new_tokens", type=int, default=512)
    ppt.add_argument("--temperature", type=float, default=0.2)
    ppt.add_argument("--top_p", type=float, default=0.9)

    # Output/debug flags
    ppt.add_argument("--include_nir_evidence", action="store_true", help="also attach NIR evidence images if present")
    ppt.add_argument("--json", action="store_true", help="force strict JSON output")
    ppt.add_argument("--show_hits", action="store_true")
    ppt.add_argument("--show_prompt", action="store_true")
    ppt.set_defaults(func=cmd_predict_text)

    # predict image (RAG -> Qwen3-VL)
    ppi = sub.add_parser("predict-image", help="Image -> RAG retrieval -> Qwen3-VL disaster prediction")
    ppi.add_argument("--db_dir", default="./data/chroma_db")
    ppi.add_argument("--collection", default="agri_global")
    ppi.add_argument("--image_path", required=True, help="target image to be predicted")
    ppi.add_argument("--query", default="请识别该农田图像中的灾害/异常类型，并结合检索证据给出理由。")
    ppi.add_argument("--top_k", type=int, default=5)
    ppi.add_argument("--evidence_k", type=int, default=6)
    ppi.add_argument("--clip_model", default="openai/clip-vit-base-patch32")

    # query image embedding options
    ppi.add_argument("--use_nir", action="store_true", help="use NIR for query embedding if available")
    ppi.add_argument("--nir_path", default=None, help="optional explicit NIR path for the query image")

    # Qwen model
    ppi.add_argument("--qwen_model", default="Qwen/Qwen3-VL-4B-Instruct",
                     help="HF repo id OR local directory path")
    ppi.add_argument("--qwen_local_root", default="./models",
                     help="if qwen_model is repo id, try local path under this root first (default ./models)")

    # Qwen runtime configs
    ppi.add_argument("--device_map", default="auto")
    ppi.add_argument("--dtype", default="auto", help="auto/float16/bfloat16...")
    ppi.add_argument("--attn_impl", default=None)
    ppi.add_argument("--max_new_tokens", type=int, default=512)
    ppi.add_argument("--temperature", type=float, default=0.2)
    ppi.add_argument("--top_p", type=float, default=0.9)

    # Output/debug flags
    ppi.add_argument("--include_nir_evidence", action="store_true")
    ppi.add_argument("--json", action="store_true")
    ppi.add_argument("--show_hits", action="store_true")
    ppi.add_argument("--show_prompt", action="store_true")
    ppi.set_defaults(func=cmd_predict_image)

    return p


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
