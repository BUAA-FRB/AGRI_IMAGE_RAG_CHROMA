# agri_rag/rag_predictor.py
from __future__ import annotations

import os
import json
import logging
from typing import Any, Dict, List, Optional, Tuple

from .embedder import ClipEmbedder
from .image_utils import open_gray, open_rgb
from .chroma_store import get_or_create_collection, query
from .prompt import build_qwen3vl_messages
from .qwen3_vl_client import Qwen3VLClient, Qwen3VLGenConfig
from .types import RetrievalHit
from .reranker import QwenTextClient, QwenTextGenConfig, rerank_hits_with_qwen

# NEW: evidence context + auto dump & montage
from . import structured_io  # NEW

logger = logging.getLogger(__name__)


def _infer_nir_from_rgb_path(rgb_path: str) -> Optional[str]:
    parts = rgb_path.replace("\\", "/")
    if "/field_images/rgb/" in parts:
        nir = parts.replace("/field_images/rgb/", "/field_images/nir/")
        nir = nir.replace("/", os.sep)
        if os.path.exists(nir):
            return os.path.abspath(nir)
    return None


def _filter_self_hits(hits: List[RetrievalHit], query_image_path: str) -> List[RetrievalHit]:
    q = os.path.abspath(query_image_path)
    out: List[RetrievalHit] = []
    for h in hits:
        p = (h.metadata or {}).get("path", "")
        try:
            if p and os.path.abspath(str(p)) == q:
                continue
        except Exception:
            pass
        out.append(h)
    return out


def _hit_uid(h: RetrievalHit) -> str:
    hid = getattr(h, "id", None)
    if hid is not None:
        return str(hid)
    md = h.metadata or {}
    return f"{md.get('type','')}|{md.get('path','')}|{md.get('bbox','')}"


def _hit_brief(h: RetrievalHit, extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    md = h.metadata or {}
    d = {
        "id": getattr(h, "id", None),
        "distance": getattr(h, "distance", None),
        "type": md.get("type", ""),
        "tile_id": md.get("tile_id", ""),
        "split": md.get("split", ""),
        "labels_present": md.get("labels_present", ""),
        "path": md.get("path", ""),
        "nir_path": md.get("nir_path", ""),  # keep
        "bbox": md.get("bbox", ""),
        "year": md.get("year", ""),
    }
    if extra:
        d.update(extra)
    return d


# -------------------------
# Verification helpers
# -------------------------
def _dedup_keep_order(xs: List[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for x in xs:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def _parse_labels_present(labels_present: Any) -> List[str]:
    if labels_present is None:
        return []
    if isinstance(labels_present, list):
        out = []
        for it in labels_present:
            if isinstance(it, str) and it.strip():
                out.append(it.strip())
        return _dedup_keep_order(out)

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
            return _dedup_keep_order(out)
    except Exception:
        pass

    for sep in [",", "，", "、", "|", ";", "/"]:
        if sep in s:
            parts = [p.strip().strip('"').strip("'") for p in s.split(sep)]
            parts = [p for p in parts if p]
            return _dedup_keep_order(parts)
    return [s]


def _union_labels_from_hit_briefs(hit_briefs: List[Dict[str, Any]]) -> List[str]:
    labs: List[str] = []
    for hb in hit_briefs or []:
        labs.extend(_parse_labels_present(hb.get("labels_present", "")))
    return _dedup_keep_order([x for x in labs if x])


def _jaccard(a: List[str], b: List[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / float(len(sa | sb))


def _verify_three_stage(dbg: Dict[str, Any], gen_raw: Optional[str] = None) -> Dict[str, Any]:
    s1_briefs = (dbg.get("stage1", {}) or {}).get("before_topk", []) or []
    if "rerank" in dbg and isinstance(dbg.get("rerank", None), dict) and dbg["rerank"].get("after_topk", None) is not None:
        s2_briefs = dbg["rerank"].get("after_topk", []) or []
        s2_source = "rerank.after_topk"
    else:
        s2_briefs = (dbg.get("final", {}) or {}).get("after_topk", []) or s1_briefs
        s2_source = "final.after_topk"

    s1_labels = _union_labels_from_hit_briefs(s1_briefs)
    s2_labels = _union_labels_from_hit_briefs(s2_briefs)

    rep: Dict[str, Any] = {
        "mode": "three_stage",
        "stage1": {"source": "stage1.before_topk", "labels_union": s1_labels, "labels_count": len(s1_labels)},
        "stage2": {"source": s2_source, "labels_union": s2_labels, "labels_count": len(s2_labels)},
        "s1_s2": {"jaccard": _jaccard(s1_labels, s2_labels), "overlap": _dedup_keep_order([x for x in s1_labels if x in set(s2_labels)])},
    }

    if gen_raw is None:
        rep["stage3"] = None
        rep["decision"] = {"status": "ok", "note": "no generation output provided; verified stage1/stage2 only"}
        return rep

    # IMPORTANT: verification should NOT trigger auto dump (avoid duplicate files)
    try:
        from .structured_io import parse_json_object, normalize_predicted_labels  # NEW parse-only
        obj = parse_json_object(gen_raw) or {}
        s3_labels = normalize_predicted_labels(obj) or []
        parse_ok = True
    except Exception:
        obj = {}
        s3_labels = []
        parse_ok = False

    s2_set = set(s2_labels)
    supported = [x for x in s3_labels if x in s2_set]
    hallucinated = [x for x in s3_labels if x not in s2_set]
    support_rate = (len(supported) / float(len(s3_labels))) if s3_labels else 0.0

    rep["stage3"] = {
        "parse_ok": parse_ok,
        "labels": s3_labels,
        "labels_count": len(s3_labels),
        "supported": supported,
        "hallucinated": hallucinated,
        "support_rate": support_rate,
        "raw_json": obj if obj else None,
    }

    if not parse_ok:
        status = "warn"
        note = "generation output JSON parse failed; cannot verify stage3 reliably"
    elif not s3_labels:
        status = "warn"
        note = "no labels extracted from generation output"
    elif len(supported) == 0:
        status = "warn"
        note = "none of generated labels are supported by stage2 evidence labels"
    elif len(hallucinated) > 0:
        status = "warn"
        note = "some generated labels are not supported by stage2 evidence labels"
    else:
        status = "ok"
        note = "all generated labels are supported by stage2 evidence labels"

    rep["decision"] = {"status": status, "note": note}
    return rep


# -------------------------
# Fusion helpers
# -------------------------
def _fuse_hits_distance(global_hits: List[RetrievalHit], patch_hits: List[RetrievalHit]) -> Tuple[List[RetrievalHit], Dict[str, Any]]:
    tagged: List[Tuple[str, RetrievalHit]] = []
    for h in global_hits:
        tagged.append(("global", h))
    for h in patch_hits:
        tagged.append(("patch", h))

    tagged.sort(key=lambda x: float(getattr(x[1], "distance", 1e9) or 1e9))
    fused = [h for _src, h in tagged]

    dbg = {"fusion_mode": "distance", "global_count": len(global_hits), "patch_count": len(patch_hits)}
    return fused, dbg


def _fuse_hits_rrf(global_hits: List[RetrievalHit], patch_hits: List[RetrievalHit], rrf_k: int = 60) -> Tuple[List[RetrievalHit], Dict[str, Any]]:
    score: Dict[str, float] = {}
    best_hit: Dict[str, RetrievalHit] = {}
    srcs: Dict[str, List[str]] = {}

    def add_list(src: str, hits: List[RetrievalHit]):
        for rank, h in enumerate(hits, start=1):
            uid = _hit_uid(h)
            score[uid] = score.get(uid, 0.0) + 1.0 / float(rrf_k + rank)
            if uid not in best_hit:
                best_hit[uid] = h
            else:
                d0 = float(getattr(best_hit[uid], "distance", 1e9) or 1e9)
                d1 = float(getattr(h, "distance", 1e9) or 1e9)
                if d1 < d0:
                    best_hit[uid] = h
            srcs.setdefault(uid, [])
            if src not in srcs[uid]:
                srcs[uid].append(src)

    add_list("global", global_hits)
    add_list("patch", patch_hits)

    items = []
    for uid, sc in score.items():
        items.append((sc, uid, best_hit[uid]))
    items.sort(key=lambda x: (-x[0], float(getattr(x[2], "distance", 1e9) or 1e9)))

    fused = [h for _sc, _uid, h in items]
    dbg = {
        "fusion_mode": "rrf",
        "rrf_k": rrf_k,
        "global_count": len(global_hits),
        "patch_count": len(patch_hits),
        "unique_count": len(items),
        "top_preview": [
            _hit_brief(h, extra={"rrf_score": sc, "sources": srcs.get(uid, [])})
            for (sc, uid, h) in items[: min(10, len(items))]
        ],
    }
    return fused, dbg


def _fuse_hits(global_hits: List[RetrievalHit], patch_hits: List[RetrievalHit], fusion_mode: str = "rrf", rrf_k: int = 60) -> Tuple[List[RetrievalHit], Dict[str, Any]]:
    if fusion_mode == "distance":
        return _fuse_hits_distance(global_hits, patch_hits)
    if fusion_mode == "rrf":
        return _fuse_hits_rrf(global_hits, patch_hits, rrf_k=rrf_k)
    raise ValueError(f"Unknown fusion_mode: {fusion_mode}")


def _maybe_rerank(
    user_query: str,
    hits: List[RetrievalHit],
    rerank_mode: str,
    rerank_top_n: int,
    rerank_client: Optional[QwenTextClient],
    rerank_gen: Optional[QwenTextGenConfig],
) -> Tuple[List[RetrievalHit], Dict[str, Any]]:
    if rerank_mode == "none":
        return hits, {}

    if rerank_mode == "qwen":
        if rerank_client is None:
            raise RuntimeError("rerank_mode='qwen' requires rerank_client (QwenTextClient)")

        reranked, dbg = rerank_hits_with_qwen(
            user_query=user_query,
            hits=hits,
            client=rerank_client,
            top_n=rerank_top_n,
            gen=rerank_gen,
        )
        return reranked, dbg

    raise ValueError(f"Unknown rerank_mode: {rerank_mode}")


def _init_rerank_client_if_needed(
    rerank_mode: str,
    rerank_model: str,
    rerank_local_root: str,
    rerank_device_map: str,
    rerank_dtype: str,
    rerank_attn_impl: Optional[str],
    rerank_force_download: bool,
    rerank_revision: Optional[str],
    rerank_hf_token: Optional[str],
) -> Optional[QwenTextClient]:
    if rerank_mode != "qwen":
        return None

    return QwenTextClient(
        model_name=rerank_model,
        local_root=rerank_local_root,
        revision=rerank_revision,
        force_download=rerank_force_download,
        hf_token=rerank_hf_token,
        device_map=rerank_device_map,
        dtype=rerank_dtype,
        attn_implementation=rerank_attn_impl,
        trust_remote_code=False,
    )


def rag_retrieve_from_text(
    user_query: str,
    db_dir: str,
    collection: str,
    clip_model: str,
    top_k: int = 8,
    rerank_mode: str = "none",
    rerank_top_n: int = 30,
    rerank_model: str = "Qwen/Qwen2.5-1.5B-Instruct",
    rerank_local_root: str = "./models",
    rerank_device_map: str = "auto",
    rerank_dtype: str = "auto",
    rerank_attn_impl: Optional[str] = None,
    rerank_force_download: bool = False,
    rerank_revision: Optional[str] = None,
    rerank_hf_token: Optional[str] = None,
    rerank_gen: Optional[QwenTextGenConfig] = None,
    use_patch_fusion: bool = False,
    patch_collection: str = "agri_patch",
    fusion_mode: str = "rrf",
    fusion_rrf_k: int = 60,
    global_top_k: Optional[int] = None,
    patch_top_k: Optional[int] = None,
    stage1_k: Optional[int] = None,
    enable_verify: bool = False,
    verify_mode: str = "three_stage",
) -> Tuple[List[RetrievalHit], Dict[str, Any]]:
    embedder = ClipEmbedder(model_name=clip_model)
    global_col = get_or_create_collection(db_dir, collection)
    patch_col = get_or_create_collection(db_dir, patch_collection) if use_patch_fusion else None

    base_stage1_k = max(top_k, rerank_top_n) if rerank_mode != "none" else top_k
    if stage1_k is not None:
        try:
            base_stage1_k = int(stage1_k)
        except Exception:
            base_stage1_k = base_stage1_k
        base_stage1_k = max(base_stage1_k, top_k)
        if rerank_mode != "none":
            base_stage1_k = max(base_stage1_k, rerank_top_n)

    qemb = embedder.embed_text(user_query)

    gk = int(global_top_k) if global_top_k is not None else base_stage1_k
    pk = int(patch_top_k) if patch_top_k is not None else base_stage1_k

    fused_keep_k = base_stage1_k
    if patch_col is not None and (global_top_k is not None or patch_top_k is not None):
        fused_keep_k = max(base_stage1_k, gk + pk)

    if patch_col is None:
        hits = query(global_col, qemb, top_k=base_stage1_k)
        fusion_dbg: Dict[str, Any] = {"use_patch_fusion": False, "global_collection": collection, "stage1_k": base_stage1_k}
    else:
        g_hits = query(global_col, qemb, top_k=gk)
        p_hits = query(patch_col, qemb, top_k=pk)
        fused, fusion_dbg = _fuse_hits(g_hits, p_hits, fusion_mode=fusion_mode, rrf_k=fusion_rrf_k)

        hits = fused[:fused_keep_k]
        fusion_dbg.update({
            "use_patch_fusion": True,
            "fusion_mode": fusion_mode,
            "global_collection": collection,
            "patch_collection": patch_collection,
            "global_query_k": gk,
            "patch_query_k": pk,
            "fused_keep_k": fused_keep_k,
        })

    dbg: Dict[str, Any] = {
        "fusion": fusion_dbg,
        "stage1": {
            "stage1_k": base_stage1_k,
            "top_k": top_k,
            "rerank_top_n": rerank_top_n,
            "global_top_k": global_top_k,
            "patch_top_k": patch_top_k,
            "before_topk": [_hit_brief(h) for h in hits[:top_k]],
        },
    }

    text_client = _init_rerank_client_if_needed(
        rerank_mode=rerank_mode,
        rerank_model=rerank_model,
        rerank_local_root=rerank_local_root,
        rerank_device_map=rerank_device_map,
        rerank_dtype=rerank_dtype,
        rerank_attn_impl=rerank_attn_impl,
        rerank_force_download=rerank_force_download,
        rerank_revision=rerank_revision,
        rerank_hf_token=rerank_hf_token,
    )

    if rerank_mode == "qwen" and rerank_gen is None:
        rerank_gen = QwenTextGenConfig(max_new_tokens=1024, temperature=0.0)

    hits2, rerank_dbg = _maybe_rerank(
        user_query=user_query,
        hits=hits,
        rerank_mode=rerank_mode,
        rerank_top_n=rerank_top_n,
        rerank_client=text_client,
        rerank_gen=rerank_gen,
    )

    if rerank_dbg:
        dbg["rerank"] = rerank_dbg
        dbg["rerank"]["rerank_model"] = rerank_model
        dbg["rerank"]["after_topk"] = [_hit_brief(h) for h in hits2[:top_k]]

    hits_final = hits2[:top_k]
    dbg["final"] = {"after_topk": [_hit_brief(h) for h in hits_final]}

    if enable_verify and verify_mode == "three_stage":
        dbg["verify"] = _verify_three_stage(dbg, gen_raw=None)

    return hits_final, dbg


def rag_retrieve_from_image(
    image_path: str,
    user_query: str,
    db_dir: str,
    collection: str,
    clip_model: str,
    top_k: int = 8,
    use_nir_for_query: bool = False,
    nir_path: Optional[str] = None,
    exclude_self: bool = True,
    rerank_mode: str = "none",
    rerank_top_n: int = 30,
    rerank_model: str = "Qwen/Qwen2.5-1.5B-Instruct",
    rerank_local_root: str = "./models",
    rerank_device_map: str = "auto",
    rerank_dtype: str = "auto",
    rerank_attn_impl: Optional[str] = None,
    rerank_force_download: bool = False,
    rerank_revision: Optional[str] = None,
    rerank_hf_token: Optional[str] = None,
    rerank_gen: Optional[QwenTextGenConfig] = None,
    use_patch_fusion: bool = False,
    patch_collection: str = "agri_patch",
    fusion_mode: str = "rrf",
    fusion_rrf_k: int = 60,
    global_top_k: Optional[int] = None,
    patch_top_k: Optional[int] = None,
    stage1_k: Optional[int] = None,
    enable_verify: bool = False,
    verify_mode: str = "three_stage",
) -> Tuple[List[RetrievalHit], Dict[str, Any]]:
    embedder = ClipEmbedder(model_name=clip_model)
    global_col = get_or_create_collection(db_dir, collection)
    patch_col = get_or_create_collection(db_dir, patch_collection) if use_patch_fusion else None

    rgb = open_rgb(image_path)
    resolved_nir_path = nir_path
    if use_nir_for_query and not resolved_nir_path:
        resolved_nir_path = _infer_nir_from_rgb_path(image_path)

    nir = None
    if use_nir_for_query and resolved_nir_path and os.path.exists(resolved_nir_path):
        nir = open_gray(resolved_nir_path)

    if nir is not None:
        qemb = embedder.embed_image_rgb_nir(rgb, nir)
    else:
        qemb = embedder.embed_image_rgb(rgb)

    base_stage1_k = max(top_k, rerank_top_n) if rerank_mode != "none" else top_k
    if exclude_self:
        base_stage1_k = max(base_stage1_k, top_k + 5)

    if stage1_k is not None:
        try:
            base_stage1_k = int(stage1_k)
        except Exception:
            base_stage1_k = base_stage1_k
        base_stage1_k = max(base_stage1_k, top_k)
        if rerank_mode != "none":
            base_stage1_k = max(base_stage1_k, rerank_top_n)
        if exclude_self:
            base_stage1_k = max(base_stage1_k, top_k + 5)

    gk = int(global_top_k) if global_top_k is not None else base_stage1_k
    pk = int(patch_top_k) if patch_top_k is not None else base_stage1_k

    fused_keep_k = base_stage1_k
    if patch_col is not None and (global_top_k is not None or patch_top_k is not None):
        fused_keep_k = max(base_stage1_k, gk + pk)

    if patch_col is None:
        hits = query(global_col, qemb, top_k=base_stage1_k)
        if exclude_self:
            hits = _filter_self_hits(hits, image_path)
        fusion_dbg: Dict[str, Any] = {"use_patch_fusion": False, "global_collection": collection, "stage1_k": base_stage1_k}
    else:
        g_hits = query(global_col, qemb, top_k=gk)
        p_hits = query(patch_col, qemb, top_k=pk)
        if exclude_self:
            g_hits = _filter_self_hits(g_hits, image_path)
            p_hits = _filter_self_hits(p_hits, image_path)

        fused, fusion_dbg = _fuse_hits(g_hits, p_hits, fusion_mode=fusion_mode, rrf_k=fusion_rrf_k)
        hits = fused[:fused_keep_k]

        fusion_dbg.update({
            "use_patch_fusion": True,
            "fusion_mode": fusion_mode,
            "global_collection": collection,
            "patch_collection": patch_collection,
            "global_query_k": gk,
            "patch_query_k": pk,
            "fused_keep_k": fused_keep_k,
        })

    dbg: Dict[str, Any] = {
        "fusion": fusion_dbg,
        "stage1": {
            "stage1_k": base_stage1_k,
            "top_k": top_k,
            "rerank_top_n": rerank_top_n,
            "global_top_k": global_top_k,
            "patch_top_k": patch_top_k,
            "before_topk": [_hit_brief(h) for h in hits[:top_k]],
        },
    }

    text_client = _init_rerank_client_if_needed(
        rerank_mode=rerank_mode,
        rerank_model=rerank_model,
        rerank_local_root=rerank_local_root,
        rerank_device_map=rerank_device_map,
        rerank_dtype=rerank_dtype,
        rerank_attn_impl=rerank_attn_impl,
        rerank_force_download=rerank_force_download,
        rerank_revision=rerank_revision,
        rerank_hf_token=rerank_hf_token,
    )

    if rerank_mode == "qwen" and rerank_gen is None:
        rerank_gen = QwenTextGenConfig(max_new_tokens=1024, temperature=0.0)

    hits2, rerank_dbg = _maybe_rerank(
        user_query=user_query,
        hits=hits,
        rerank_mode=rerank_mode,
        rerank_top_n=rerank_top_n,
        rerank_client=text_client,
        rerank_gen=rerank_gen,
    )

    if rerank_dbg:
        dbg["rerank"] = rerank_dbg
        dbg["rerank"]["rerank_model"] = rerank_model
        dbg["rerank"]["after_topk"] = [_hit_brief(h) for h in hits2[:top_k]]

    hits_final = hits2[:top_k]
    dbg["final"] = {"after_topk": [_hit_brief(h) for h in hits_final]}

    if enable_verify and verify_mode == "three_stage":
        dbg["verify"] = _verify_three_stage(dbg, gen_raw=None)

    return hits_final, dbg


def rag_predict_from_text(
    user_query: str,
    db_dir: str,
    collection: str,
    clip_model: str,
    qwen_model: str,
    top_k: int = 5,
    evidence_k: int = 6,
    include_nir_evidence: bool = False,
    output_json: bool = False,
    qwen_device_map: str = "auto",
    qwen_dtype: str = "auto",
    qwen_attn_impl: Optional[str] = None,
    gen: Optional[Qwen3VLGenConfig] = None,
    rerank_mode: str = "none",
    rerank_top_n: int = 30,
    rerank_model: str = "Qwen/Qwen2.5-1.5B-Instruct",
    rerank_local_root: str = "./models",
    rerank_device_map: str = "auto",
    rerank_dtype: str = "auto",
    rerank_attn_impl: Optional[str] = None,
    rerank_force_download: bool = False,
    rerank_revision: Optional[str] = None,
    rerank_hf_token: Optional[str] = None,
    rerank_gen: Optional[QwenTextGenConfig] = None,
    use_patch_fusion: bool = False,
    patch_collection: str = "agri_patch",
    fusion_mode: str = "rrf",
    fusion_rrf_k: int = 60,
    global_top_k: Optional[int] = None,
    patch_top_k: Optional[int] = None,
    stage1_k: Optional[int] = None,
    enable_verify: bool = False,
    verify_mode: str = "three_stage",
) -> Tuple[str, List[RetrievalHit], List[Dict[str, Any]], Dict[str, Any]]:
    hits, dbg = rag_retrieve_from_text(
        user_query=user_query,
        db_dir=db_dir,
        collection=collection,
        clip_model=clip_model,
        top_k=top_k,
        rerank_mode=rerank_mode,
        rerank_top_n=rerank_top_n,
        rerank_model=rerank_model,
        rerank_local_root=rerank_local_root,
        rerank_device_map=rerank_device_map,
        rerank_dtype=rerank_dtype,
        rerank_attn_impl=rerank_attn_impl,
        rerank_force_download=rerank_force_download,
        rerank_revision=rerank_revision,
        rerank_hf_token=rerank_hf_token,
        rerank_gen=rerank_gen,
        use_patch_fusion=use_patch_fusion,
        patch_collection=patch_collection,
        fusion_mode=fusion_mode,
        fusion_rrf_k=fusion_rrf_k,
        global_top_k=global_top_k,
        patch_top_k=patch_top_k,
        stage1_k=stage1_k,
        enable_verify=enable_verify,
        verify_mode=verify_mode,
    )

    messages = build_qwen3vl_messages(
        user_query=user_query,
        hits=hits,
        query_image_path=None,
        max_hits=evidence_k,
        include_nir_evidence=include_nir_evidence,
        output_json=output_json,
    )

    # NEW: provide evidence mapping E1..Ek (no query image for text mode)
    try:
        structured_io.set_evidence_context_from_hits(
            hits=hits,
            query_image_path=None,
            max_hits=evidence_k,
            include_nir_evidence=include_nir_evidence,
        )
    except Exception:
        pass

    vl_client = Qwen3VLClient(
        model_name=qwen_model,
        device_map=qwen_device_map,
        dtype=qwen_dtype,
        attn_implementation=qwen_attn_impl,
    )
    out = vl_client.chat(messages, gen=gen)

    if enable_verify and verify_mode == "three_stage":
        dbg["verify"] = _verify_three_stage(dbg, gen_raw=out)

    # NEW: trigger final JSON dump + evidence assets + montage (if any)
    try:
        structured_io.extract_json_object(out)
    except Exception:
        pass

    return out, hits, messages, dbg


def rag_predict_from_image(
    image_path: str,
    user_query: str,
    db_dir: str,
    collection: str,
    clip_model: str,
    qwen_model: str,
    top_k: int = 5,
    evidence_k: int = 6,
    use_nir_for_query: bool = False,
    nir_path: Optional[str] = None,
    include_nir_evidence: bool = False,
    output_json: bool = False,
    qwen_device_map: str = "auto",
    qwen_dtype: str = "auto",
    qwen_attn_impl: Optional[str] = None,
    gen: Optional[Qwen3VLGenConfig] = None,
    rerank_mode: str = "none",
    rerank_top_n: int = 30,
    exclude_self: bool = True,
    rerank_model: str = "Qwen/Qwen2.5-1.5B-Instruct",
    rerank_local_root: str = "./models",
    rerank_device_map: str = "auto",
    rerank_dtype: str = "auto",
    rerank_attn_impl: Optional[str] = None,
    rerank_force_download: bool = False,
    rerank_revision: Optional[str] = None,
    rerank_hf_token: Optional[str] = None,
    rerank_gen: Optional[QwenTextGenConfig] = None,
    use_patch_fusion: bool = False,
    patch_collection: str = "agri_patch",
    fusion_mode: str = "rrf",
    fusion_rrf_k: int = 60,
    global_top_k: Optional[int] = None,
    patch_top_k: Optional[int] = None,
    stage1_k: Optional[int] = None,
    enable_verify: bool = False,
    verify_mode: str = "three_stage",
) -> Tuple[str, List[RetrievalHit], List[Dict[str, Any]], Dict[str, Any]]:
    hits, dbg = rag_retrieve_from_image(
        image_path=image_path,
        user_query=user_query,
        db_dir=db_dir,
        collection=collection,
        clip_model=clip_model,
        top_k=top_k,
        use_nir_for_query=use_nir_for_query,
        nir_path=nir_path,
        exclude_self=exclude_self,
        rerank_mode=rerank_mode,
        rerank_top_n=rerank_top_n,
        rerank_model=rerank_model,
        rerank_local_root=rerank_local_root,
        rerank_device_map=rerank_device_map,
        rerank_dtype=rerank_dtype,
        rerank_attn_impl=rerank_attn_impl,
        rerank_force_download=rerank_force_download,
        rerank_revision=rerank_revision,
        rerank_hf_token=rerank_hf_token,
        rerank_gen=rerank_gen,
        use_patch_fusion=use_patch_fusion,
        patch_collection=patch_collection,
        fusion_mode=fusion_mode,
        fusion_rrf_k=fusion_rrf_k,
        global_top_k=global_top_k,
        patch_top_k=patch_top_k,
        stage1_k=stage1_k,
        enable_verify=enable_verify,
        verify_mode=verify_mode,
    )

    messages = build_qwen3vl_messages(
        user_query=user_query,
        hits=hits,
        query_image_path=image_path,
        max_hits=evidence_k,
        include_nir_evidence=include_nir_evidence,
        output_json=output_json,
    )

    # NEW: provide evidence mapping E1..Ek + query image path
    try:
        structured_io.set_evidence_context_from_hits(
            hits=hits,
            query_image_path=image_path,
            max_hits=evidence_k,
            include_nir_evidence=include_nir_evidence,
        )
    except Exception:
        pass

    vl_client = Qwen3VLClient(
        model_name=qwen_model,
        device_map=qwen_device_map,
        dtype=qwen_dtype,
        attn_implementation=qwen_attn_impl,
    )
    out = vl_client.chat(messages, gen=gen)

    if enable_verify and verify_mode == "three_stage":
        dbg["verify"] = _verify_three_stage(dbg, gen_raw=out)

    # NEW: trigger final JSON dump + evidence assets + montage
    try:
        structured_io.extract_json_object(out)
    except Exception:
        pass

    return out, hits, messages, dbg
