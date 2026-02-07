# agri_rag/rag_predictor.py
from __future__ import annotations

import os
import logging
from typing import Any, Dict, List, Optional, Tuple

from .embedder import ClipEmbedder
from .image_utils import open_gray, open_rgb
from .chroma_store import get_or_create_collection, query
from .prompt import build_qwen3vl_messages
from .qwen3_vl_client import Qwen3VLClient, Qwen3VLGenConfig
from .types import RetrievalHit

from .reranker import QwenTextClient, QwenTextGenConfig, rerank_hits_with_qwen

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


def _hit_brief(h: RetrievalHit) -> Dict[str, Any]:
    md = h.metadata or {}
    return {
        "id": getattr(h, "id", None),
        "distance": getattr(h, "distance", None),
        "tile_id": md.get("tile_id", ""),
        "split": md.get("split", ""),
        "labels_present": md.get("labels_present", ""),
        "path": md.get("path", ""),
    }


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
) -> Tuple[List[RetrievalHit], Dict[str, Any]]:
    embedder = ClipEmbedder(model_name=clip_model)
    col = get_or_create_collection(db_dir, collection)

    stage1_k = max(top_k, rerank_top_n) if rerank_mode != "none" else top_k
    qemb = embedder.embed_text(user_query)
    hits = query(col, qemb, top_k=stage1_k)

    dbg: Dict[str, Any] = {
        "stage1": {
            "stage1_k": stage1_k,
            "top_k": top_k,
            "rerank_top_n": rerank_top_n,
            "before_topk": [_hit_brief(h) for h in hits[:top_k]],
        }
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

    hits, rerank_dbg = _maybe_rerank(
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
        dbg["rerank"]["after_topk"] = [_hit_brief(h) for h in hits[:top_k]]

    hits = hits[:top_k]
    return hits, dbg


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
) -> Tuple[List[RetrievalHit], Dict[str, Any]]:
    embedder = ClipEmbedder(model_name=clip_model)
    col = get_or_create_collection(db_dir, collection)

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

    stage1_k = max(top_k, rerank_top_n) if rerank_mode != "none" else top_k
    if exclude_self:
        stage1_k = max(stage1_k, top_k + 5)

    hits = query(col, qemb, top_k=stage1_k)
    if exclude_self:
        hits = _filter_self_hits(hits, image_path)

    dbg: Dict[str, Any] = {
        "stage1": {
            "stage1_k": stage1_k,
            "top_k": top_k,
            "rerank_top_n": rerank_top_n,
            "before_topk": [_hit_brief(h) for h in hits[:top_k]],
        }
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

    hits, rerank_dbg = _maybe_rerank(
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
        dbg["rerank"]["after_topk"] = [_hit_brief(h) for h in hits[:top_k]]

    hits = hits[:top_k]
    return hits, dbg


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
    )

    messages = build_qwen3vl_messages(
        user_query=user_query,
        hits=hits,
        query_image_path=None,
        max_hits=evidence_k,
        include_nir_evidence=include_nir_evidence,
        output_json=output_json,
    )

    vl_client = Qwen3VLClient(
        model_name=qwen_model,
        device_map=qwen_device_map,
        dtype=qwen_dtype,
        attn_implementation=qwen_attn_impl,
    )
    out = vl_client.chat(messages, gen=gen)
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
    )

    messages = build_qwen3vl_messages(
        user_query=user_query,
        hits=hits,
        query_image_path=image_path,
        max_hits=evidence_k,
        include_nir_evidence=include_nir_evidence,
        output_json=output_json,
    )

    vl_client = Qwen3VLClient(
        model_name=qwen_model,
        device_map=qwen_device_map,
        dtype=qwen_dtype,
        attn_implementation=qwen_attn_impl,
    )
    out = vl_client.chat(messages, gen=gen)
    return out, hits, messages, dbg
