# cli.py
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
from .qwen3_vl_client import Qwen3VLGenConfig, Qwen3VLClient

# ---- Agent & Report imports ----
from .agents import AgentRouter, PlantingAnomalyAgent
from .report import MultimodalReportGenerator


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


# ============================================================
# Planting Anomaly Detection (Agent Pipeline)
# ============================================================
def cmd_planting_detect(args: argparse.Namespace) -> None:
    """
    播种异常检测 Agent 完整流程：
    1. 通用RAG预测（复用现有 predict-image 逻辑）
    2. Agent路由 → PlantingAnomalyAgent 专项分析
    3. 可选：生成多模态可视化报告
    """
    import logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")

    qwen_model = _maybe_resolve_qwen_local_path(args.qwen_model, args.qwen_local_root)

    gen = Qwen3VLGenConfig(
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        do_sample=None,
    )

    # ---- Step 1: 通用RAG预测 ----
    print("=" * 70)
    print("Step 1: 通用RAG预测")
    print("=" * 70)

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
        output_json=True,  # Agent需要JSON输出
        qwen_device_map=args.device_map,
        qwen_dtype=args.dtype,
        qwen_attn_impl=args.attn_impl,
        gen=gen,
    )

    print(f"\n通用RAG原始输出:\n{out}\n")

    # 解析通用预测JSON
    general_prediction = _safe_parse_json(out)
    if general_prediction is None:
        print("[WARNING] 通用RAG输出无法解析为JSON，将使用文本匹配模式")
        general_prediction = {"conclusion": out}

    # ---- Step 2: 获取patch级检索结果（可选） ----
    patch_hits = None
    if args.patch_collection:
        print("=" * 70)
        print("Step 2: Patch级检索")
        print("=" * 70)
        try:
            embedder = ClipEmbedder(model_name=args.clip_model)
            patch_col = get_or_create_collection(args.db_dir, args.patch_collection)
            rgb = open_rgb(args.image_path)

            nir_path = args.nir_path
            if args.use_nir and not nir_path:
                nir_path = _infer_nir_from_rgb_path(args.image_path)
            nir = open_gray(nir_path) if args.use_nir and nir_path and os.path.exists(nir_path) else None

            if nir is not None:
                qemb = embedder.embed_image_rgb_nir(rgb, nir)
            else:
                qemb = embedder.embed_image_rgb(rgb)

            patch_hits = query(patch_col, qemb, top_k=args.patch_top_k)
            print(f"Patch检索完成: {len(patch_hits)} 条结果")
        except Exception as e:
            print(f"[WARNING] Patch检索失败: {e}")
            patch_hits = None

    # ---- Step 3: Agent路由分发 ----
    print("\n" + "=" * 70)
    print("Step 3: 播种异常Agent分析")
    print("=" * 70)

    # 构建Qwen client用于Phase 2二次推理（可选）
    qwen_client = None
    if args.enable_vlm_reasoning:
        qwen_client = Qwen3VLClient(
            model_name=qwen_model,
            device_map=args.device_map,
            dtype=args.dtype,
            attn_implementation=args.attn_impl,
        )

    router = AgentRouter()
    reports = router.route(
        image_path=args.image_path,
        general_prediction=general_prediction,
        hits=hits,
        patch_hits=patch_hits,
        qwen_client=qwen_client,
        force_agents=["planting_anomaly_agent"] if args.force else None,
    )

    if not reports:
        print("\n[INFO] 未检测到播种异常，Agent未激活。")
        print("通用预测结果:")
        print(json.dumps(general_prediction, ensure_ascii=False, indent=2))
        return

    # ---- Step 4: 输出结果 ----
    for report in reports:
        print(f"\n{'=' * 70}")
        print(f"Agent: {report.agent_name}")
        print(f"{'=' * 70}")
        print(f"摘要: {report.summary}")
        print(f"整体严重度: {report.overall_severity.value}")
        print(f"整体置信度: {report.overall_confidence:.2f}")
        print(f"紧急程度: {report.urgency.value}")

        print(f"\n--- 检测结果 ---")
        for d in report.detected_anomalies:
            type_cn = {"double_plant": "重复播种", "planter_skip": "跳播/漏播"}.get(d.label, d.label)
            status = "✓ 检出" if d.detected else "✗ 未检出"
            print(f"  {type_cn}: {status}")
            if d.detected:
                print(f"    置信度: {d.confidence:.2f}")
                print(f"    严重度: {d.severity.value}")
                print(f"    面积占比: {d.affected_area_ratio:.1%}")
                print(f"    分布模式: {d.distribution.value}")
                if d.visual_evidence:
                    print(f"    视觉描述: {d.visual_evidence}")

        if report.spatial_summary:
            sp = report.spatial_summary
            print(f"\n--- 空间分析 ---")
            print(f"  异常区域: {sp.total_anomaly_regions} 处")
            print(f"  受影响面积: {sp.total_affected_ratio:.1%}")
            print(f"  主要位置: {sp.primary_location}")
            print(f"  分布描述: {sp.pattern_description}")

        if report.recommendations:
            print(f"\n--- 补救建议 ({len(report.recommendations)} 条) ---")
            for i, rec in enumerate(report.recommendations, 1):
                print(f"  {i}. [{rec.urgency.value}] {rec.action}")
                print(f"     针对: {rec.target_anomaly} | 区域: {rec.target_area}")
                print(f"     预期: {rec.expected_effect}")
                if rec.notes:
                    print(f"     备注: {rec.notes}")

        # ---- Step 5: 保存JSON报告 ----
        output_dir = args.output_dir
        os.makedirs(output_dir, exist_ok=True)

        json_path = os.path.join(output_dir, "planting_report.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(report.to_dict(), f, ensure_ascii=False, indent=2)
        print(f"\n✓ JSON报告已保存: {json_path}")

        # ---- Step 6: 生成HTML可视化报告（可选） ----
        if args.generate_report:
            print(f"\n--- 生成可视化报告 ---")
            try:
                gen_report = MultimodalReportGenerator()
                html_path = gen_report.generate(
                    report=report,
                    hits=hits,
                    output_dir=output_dir,
                    title=f"播种异常分析 - {os.path.basename(args.image_path)}",
                    embed_images=True,
                )
                print(f"✓ HTML报告已保存: {html_path}")
            except Exception as e:
                print(f"[WARNING] HTML报告生成失败: {e}")
                import traceback
                traceback.print_exc()


def _safe_parse_json(text: str) -> Optional[Dict[str, Any]]:
    """安全解析JSON，兼容代码块包裹"""
    if not text:
        return None
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            try:
                return json.loads(text[start:end + 1])
            except json.JSONDecodeError:
                pass
    return None


# =========================
# Argument Parser
# =========================
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="agri_rag")
    sub = p.add_subparsers(dest="cmd", required=True)

    # ---- index ----
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

    # ---- query text ----
    pqt = sub.add_parser("query-text", help="Text -> image retrieval (CLIP text encoder)")
    pqt.add_argument("--db_dir", default="./data/chroma_db")
    pqt.add_argument("--collection", default="agri_global")
    pqt.add_argument("--query", required=True)
    pqt.add_argument("--top_k", type=int, default=5)
    pqt.add_argument("--build_prompt", action="store_true", help="also output a RAG prompt for LLMs")
    pqt.add_argument("--clip_model", default="openai/clip-vit-base-patch32")
    pqt.set_defaults(func=cmd_query_text)

    # ---- query image ----
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

    # ---- predict text ----
    ppt = sub.add_parser("predict-text", help="Text -> RAG retrieval -> Qwen3-VL disaster prediction")
    ppt.add_argument("--db_dir", default="./data/chroma_db")
    ppt.add_argument("--collection", default="agri_global")
    ppt.add_argument("--query", required=True, help="user question / description")
    ppt.add_argument("--top_k", type=int, default=5, help="retrieval top_k")
    ppt.add_argument("--evidence_k", type=int, default=6, help="how many retrieved items to attach as evidence images")
    ppt.add_argument("--clip_model", default="openai/clip-vit-base-patch32")
    ppt.add_argument("--qwen_model", default="Qwen/Qwen3-VL-4B-Instruct",
                     help="HF repo id (e.g., Qwen/Qwen3-VL-4B-Instruct) OR local directory path")
    ppt.add_argument("--qwen_local_root", default="./models",
                     help="if qwen_model is repo id, try local path under this root first (default ./models)")
    ppt.add_argument("--device_map", default="auto")
    ppt.add_argument("--dtype", default="auto", help="auto/float16/bfloat16...")
    ppt.add_argument("--attn_impl", default=None, help="e.g. flash_attention_2 / sdpa")
    ppt.add_argument("--max_new_tokens", type=int, default=512)
    ppt.add_argument("--temperature", type=float, default=0.2)
    ppt.add_argument("--top_p", type=float, default=0.9)
    ppt.add_argument("--include_nir_evidence", action="store_true", help="also attach NIR evidence images if present")
    ppt.add_argument("--json", action="store_true", help="force strict JSON output")
    ppt.add_argument("--show_hits", action="store_true")
    ppt.add_argument("--show_prompt", action="store_true")
    ppt.set_defaults(func=cmd_predict_text)

    # ---- predict image ----
    ppi = sub.add_parser("predict-image", help="Image -> RAG retrieval -> Qwen3-VL disaster prediction")
    ppi.add_argument("--db_dir", default="./data/chroma_db")
    ppi.add_argument("--collection", default="agri_global")
    ppi.add_argument("--image_path", required=True, help="target image to be predicted")
    ppi.add_argument("--query", default="请识别该农田图像中的灾害/异常类型，并结合检索证据给出理由。")
    ppi.add_argument("--top_k", type=int, default=5)
    ppi.add_argument("--evidence_k", type=int, default=6)
    ppi.add_argument("--clip_model", default="openai/clip-vit-base-patch32")
    ppi.add_argument("--use_nir", action="store_true", help="use NIR for query embedding if available")
    ppi.add_argument("--nir_path", default=None, help="optional explicit NIR path for the query image")
    ppi.add_argument("--qwen_model", default="Qwen/Qwen3-VL-4B-Instruct",
                     help="HF repo id OR local directory path")
    ppi.add_argument("--qwen_local_root", default="./models",
                     help="if qwen_model is repo id, try local path under this root first (default ./models)")
    ppi.add_argument("--device_map", default="auto")
    ppi.add_argument("--dtype", default="auto", help="auto/float16/bfloat16...")
    ppi.add_argument("--attn_impl", default=None)
    ppi.add_argument("--max_new_tokens", type=int, default=512)
    ppi.add_argument("--temperature", type=float, default=0.2)
    ppi.add_argument("--top_p", type=float, default=0.9)
    ppi.add_argument("--include_nir_evidence", action="store_true")
    ppi.add_argument("--json", action="store_true")
    ppi.add_argument("--show_hits", action="store_true")
    ppi.add_argument("--show_prompt", action="store_true")
    ppi.set_defaults(func=cmd_predict_image)

    # ============================================================
    # ---- planting-detect (NEW: Agent Pipeline) ----
    # ============================================================
    ppd = sub.add_parser("planting-detect",
                         help="Image -> RAG -> PlantingAnomalyAgent -> structured report + HTML visualization")
    ppd.add_argument("--image_path", required=True, help="target image to be analyzed")
    ppd.add_argument("--query", default="请识别该农田图像中的播种异常（重复播种/跳播漏播），并结合检索证据给出详细分析。")

    # Retrieval
    ppd.add_argument("--db_dir", default="./data/chroma_db")
    ppd.add_argument("--collection", default="agri_global", help="global embedding collection")
    ppd.add_argument("--patch_collection", default="", help="patch embedding collection (empty to skip patch retrieval)")
    ppd.add_argument("--top_k", type=int, default=8, help="retrieval top_k (global)")
    ppd.add_argument("--patch_top_k", type=int, default=15, help="retrieval top_k (patch)")
    ppd.add_argument("--evidence_k", type=int, default=6, help="evidence images to attach")
    ppd.add_argument("--clip_model", default="openai/clip-vit-base-patch32")

    # NIR options
    ppd.add_argument("--use_nir", action="store_true")
    ppd.add_argument("--nir_path", default=None)
    ppd.add_argument("--include_nir_evidence", action="store_true")

    # Qwen model
    ppd.add_argument("--qwen_model", default="Qwen/Qwen3-VL-4B-Instruct")
    ppd.add_argument("--qwen_local_root", default="./models")
    ppd.add_argument("--device_map", default="auto")
    ppd.add_argument("--dtype", default="auto")
    ppd.add_argument("--attn_impl", default=None)
    ppd.add_argument("--max_new_tokens", type=int, default=512)
    ppd.add_argument("--temperature", type=float, default=0.2)
    ppd.add_argument("--top_p", type=float, default=0.9)

    # Agent options
    ppd.add_argument("--force", action="store_true",
                     help="force activate PlantingAnomalyAgent even if general prediction doesn't detect planting issues")
    ppd.add_argument("--enable_vlm_reasoning", action="store_true",
                     help="enable Phase 2: VLM secondary reasoning with planting-specific prompt")

    # Output options
    ppd.add_argument("--output_dir", default="./output/planting", help="output directory for reports")
    ppd.add_argument("--generate_report", action="store_true",
                     help="generate HTML visualization report")
    ppd.set_defaults(func=cmd_planting_detect)

    return p


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
