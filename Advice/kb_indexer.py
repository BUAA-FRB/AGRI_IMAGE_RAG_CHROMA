# Advice/kb_indexer.py
from __future__ import annotations

import json
import os
from dataclasses import asdict
from typing import Any, Dict, List, Optional

from .kb_loader import load_knowledge
from .text_embedder import TextEmbedder
from .chroma_store import get_or_create_collection, reset_collection, upsert


def _ensure_dir(p: str) -> None:
    if not p:
        return
    os.makedirs(p, exist_ok=True)


def _write_json(p: str, obj: Any) -> None:
    _ensure_dir(os.path.dirname(p) or ".")
    with open(p, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def _append_jsonl(p: str, obj: Dict[str, Any]) -> None:
    _ensure_dir(os.path.dirname(p) or ".")
    with open(p, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def _meta_pack(meta: Dict[str, Any]) -> Dict[str, Any]:
    """
    Chroma metadata 必须是标量类型（str/int/float/bool）。
    tags -> tags_s 用 | 拼接；原 tags 列表不直接入库（避免 list 报错）。
    """
    tags = meta.get("tags") or []
    if isinstance(tags, str):
        tags = [tags]
    tags_s = "|".join([str(t) for t in tags if str(t).strip()])[:500]

    out = dict(meta)
    out["tags_s"] = tags_s
    out["title"] = str(meta.get("title") or "")
    out["section"] = str(meta.get("section") or "")
    out["source"] = str(meta.get("source") or "")
    out["path"] = str(meta.get("path") or "")
    out["updated"] = str(meta.get("updated") or "")
    out.pop("tags", None)
    return out


def _cuda_available() -> bool:
    try:
        import torch  # type: ignore
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _maybe_empty_cuda_cache():
    try:
        import torch  # type: ignore
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def _embed_with_oom_retry(
    embedder: TextEmbedder,
    texts: List[str],
    try_batch: int,
) -> List[List[float]]:
    """
    外层 OOM retry：控制一次喂给 embed_texts 的数量。
    """
    if not texts:
        return []

    outs: List[List[float]] = []
    i = 0
    n = len(texts)
    bs = max(1, int(try_batch))

    while i < n:
        cur_bs = min(bs, n - i)
        chunk = texts[i:i + cur_bs]
        try:
            outs.extend(embedder.embed_texts(chunk))
            i += cur_bs
        except RuntimeError as e:
            msg = str(e).lower()
            if ("out of memory" in msg) or ("cuda" in msg and "memory" in msg):
                _maybe_empty_cuda_cache()
                if bs <= 1:
                    raise
                bs = max(1, bs // 2)
                continue
            raise
    return outs


def build_kb_chroma(
    kb_dir: str,
    db_dir: str,
    collection: str,
    embed_model: str,
    max_chars: int = 900,
    overlap: int = 160,
    batch_size: int = 64,
    reset: bool = False,
    # === embedding 配置（适配 4050 6GB 默认） ===
    embed_max_length: Optional[int] = None,
    embed_batch_size: Optional[int] = None,
    force_device: Optional[str] = None,
    empty_cuda_cache_each_batch: bool = True,
    # === 新增：人类可视化索引输出 ===
    kb_index_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """
    将 knowledge 文档切 chunk 后写入 Chroma，并可选输出 kb_index 可读索引：
      kb_index_dir/
        manifest.json
        chunks.jsonl
        docs_index.json
    """
    metric = "cosine"

    if reset:
        reset_collection(db_dir, collection, metric=metric)

    col = get_or_create_collection(db_dir, collection, metric=metric)

    use_cuda = _cuda_available()
    device = force_device or ("cuda" if use_cuda else "cpu")

    if embed_max_length is None:
        embed_max_length = 384 if use_cuda else 512
    if embed_batch_size is None:
        embed_batch_size = 12 if use_cuda else 32

    emb = TextEmbedder(
        model_name_or_path=embed_model,
        local_files_only=True,
        device=device,
        max_length=int(embed_max_length),
        batch_size=int(embed_batch_size),
    )

    # === kb_index 可读索引 ===
    chunks_jsonl = None
    docs_index_path = None
    manifest_path = None
    docs_agg: Dict[str, Dict[str, Any]] = {}

    if kb_index_dir:
        _ensure_dir(kb_index_dir)
        chunks_jsonl = os.path.join(kb_index_dir, "chunks.jsonl")
        docs_index_path = os.path.join(kb_index_dir, "docs_index.json")
        manifest_path = os.path.join(kb_index_dir, "manifest.json")
        # 重建时清空 jsonl（避免重复追加）
        if reset and os.path.exists(chunks_jsonl):
            os.remove(chunks_jsonl)

    ids: List[str] = []
    docs: List[str] = []
    metas: List[Dict[str, Any]] = []

    total = 0
    written = 0

    def flush():
        nonlocal ids, docs, metas, written
        if not ids:
            return

        vecs = _embed_with_oom_retry(emb, docs, try_batch=len(docs))

        # ✅ 修复：upsert 参数顺序必须是 metadatas=..., documents=...
        upsert(
            col,
            ids=ids,
            embeddings=vecs,
            metadatas=metas,
            documents=docs,
        )
        written += len(ids)

        ids, docs, metas = [], [], []
        if use_cuda and empty_cuda_cache_each_batch:
            _maybe_empty_cuda_cache()

    for chunk in load_knowledge(kb_dir, max_chars=max_chars, overlap=overlap):
        total += 1
        kid = chunk.kid
        text = chunk.text
        meta_raw = chunk.meta or {}

        ids.append(kid)
        docs.append(text)
        metas.append(_meta_pack(meta_raw))

        # 写 chunks.jsonl（可读）
        if chunks_jsonl:
            tags = meta_raw.get("tags") or []
            if isinstance(tags, str):
                tags = [tags]
            doc_key = str(meta_raw.get("path") or meta_raw.get("source") or meta_raw.get("title") or "unknown")

            _append_jsonl(chunks_jsonl, {
                "kid": kid,
                "doc_key": doc_key,
                "title": str(meta_raw.get("title") or ""),
                "section": str(meta_raw.get("section") or ""),
                "phase": str(meta_raw.get("phase") or ""),  # 如果你的 loader/chunker 有 phase 字段，会自动带上
                "tags": [str(t) for t in tags],
                "source": str(meta_raw.get("source") or ""),
                "path": str(meta_raw.get("path") or ""),
                "updated": str(meta_raw.get("updated") or ""),
                "n_chars": len(text or ""),
                "text_preview": (text or "")[:240],
            })

            # docs 聚合
            ent = docs_agg.setdefault(doc_key, {
                "doc_key": doc_key,
                "title": str(meta_raw.get("title") or ""),
                "source": str(meta_raw.get("source") or ""),
                "path": str(meta_raw.get("path") or ""),
                "tags": set(),
                "phases": set(),
                "chunks": 0,
            })
            ent["chunks"] += 1
            for t in tags:
                if str(t).strip():
                    ent["tags"].add(str(t))
            ph = str(meta_raw.get("phase") or "")
            if ph.strip():
                ent["phases"].add(ph)

        if len(ids) >= batch_size:
            flush()

    flush()

    # 写 docs_index / manifest
    if docs_index_path:
        out_docs = []
        for _, v in docs_agg.items():
            out_docs.append({
                "doc_key": v["doc_key"],
                "title": v["title"],
                "source": v["source"],
                "path": v["path"],
                "tags": sorted(list(v["tags"])),
                "phases": sorted(list(v["phases"])),
                "chunks": int(v["chunks"]),
            })
        out_docs.sort(key=lambda x: (-x["chunks"], x["doc_key"]))
        _write_json(docs_index_path, {"docs": out_docs})

    try:
        cnt = col.count()
    except Exception:
        cnt = None

    info = {
        "db_dir": db_dir,
        "collection": collection,
        "metric": metric,
        "embed_model": embed_model,
        "device": device,
        "embed_max_length": embed_max_length,
        "embed_batch_size": embed_batch_size,
        "flush_batch_size": batch_size,
        "items_in_collection": cnt,
        "processed_chunks": total,
        "written_chunks": written,
        "kb_index_dir": kb_index_dir or "",
    }

    if manifest_path:
        _write_json(manifest_path, info)

    return info
