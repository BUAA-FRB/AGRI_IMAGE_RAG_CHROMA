from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

from .embedder import ClipEmbedder
from .image_utils import open_gray, open_rgb
from .chroma_store import get_or_create_collection, query
from .prompt import build_qwen3vl_messages
from .qwen3_vl_client import Qwen3VLClient, Qwen3VLGenConfig
from .types import RetrievalHit


def _infer_nir_from_rgb_path(rgb_path: str) -> Optional[str]:
    # Keep the same heuristic style as cli.py
    parts = rgb_path.replace("\\", "/")
    if "/field_images/rgb/" in parts:
        nir = parts.replace("/field_images/rgb/", "/field_images/nir/")
        nir = nir.replace("/", os.sep)
        if os.path.exists(nir):
            return os.path.abspath(nir)
    return None


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
) -> Tuple[str, List[RetrievalHit], List[Dict[str, Any]]]:
    """
    Text -> retrieve top_k -> build multimodal messages (attach evidence images) -> Qwen3-VL -> prediction text
    Returns: (model_output, hits, messages)
    """
    embedder = ClipEmbedder(model_name=clip_model)
    col = get_or_create_collection(db_dir, collection)

    qemb = embedder.embed_text(user_query)
    hits = query(col, qemb, top_k=top_k)

    messages = build_qwen3vl_messages(
        user_query=user_query,
        hits=hits,
        query_image_path=None,
        max_hits=evidence_k,
        include_nir_evidence=include_nir_evidence,
        output_json=output_json,
    )

    client = Qwen3VLClient(
        model_name=qwen_model,
        device_map=qwen_device_map,
        dtype=qwen_dtype,
        attn_implementation=qwen_attn_impl,
    )
    out = client.chat(messages, gen=gen)
    return out, hits, messages


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
) -> Tuple[str, List[RetrievalHit], List[Dict[str, Any]]]:
    """
    Image -> retrieve top_k -> build multimodal messages (attach query image + evidence images) -> Qwen3-VL
    Returns: (model_output, hits, messages)
    """
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

    hits = query(col, qemb, top_k=top_k)

    messages = build_qwen3vl_messages(
        user_query=user_query,
        hits=hits,
        query_image_path=image_path,
        max_hits=evidence_k,
        include_nir_evidence=include_nir_evidence,
        output_json=output_json,
    )

    client = Qwen3VLClient(
        model_name=qwen_model,
        device_map=qwen_device_map,
        dtype=qwen_dtype,
        attn_implementation=qwen_attn_impl,
    )
    out = client.chat(messages, gen=gen)
    return out, hits, messages
