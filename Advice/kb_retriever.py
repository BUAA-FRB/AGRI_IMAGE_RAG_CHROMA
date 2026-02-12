# Advice/kb_retriever.py
from __future__ import annotations

from typing import Any, Dict, List, Optional

from .text_embedder import TextEmbedder
from .chroma_store import get_or_create_collection, query


def _cuda_available() -> bool:
    try:
        import torch  # type: ignore
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _tags_from_meta(meta: Dict[str, Any]) -> List[str]:
    tags_s = str(meta.get("tags_s") or "")
    if not tags_s:
        return []
    return [t.strip().lower() for t in tags_s.split("|") if t.strip()]


class Retriever:
    def __init__(self, db_dir: str, collection: str, embed_model: str):
        self.db_dir = db_dir
        self.collection_name = collection
        self.col = get_or_create_collection(db_dir, collection, metric="cosine")

        use_cuda = _cuda_available()
        device = "cuda" if use_cuda else "cpu"
        max_length = 384 if use_cuda else 512
        batch_size = 16 if use_cuda else 32

        self.emb = TextEmbedder(
            model_name_or_path=embed_model,
            local_files_only=True,
            device=device,
            max_length=max_length,
            batch_size=batch_size,
        )

    def search(
        self,
        query_text: str,
        top_k: int = 6,
        query_tags: Optional[List[str]] = None,
        tag_boost: float = 0.12,
    ) -> List[Dict[str, Any]]:
        q_emb = self.emb.embed_texts([query_text])[0]

        fetch_k = int(top_k)
        if query_tags:
            fetch_k = max(fetch_k, top_k * 4)

        res = query(self.col, q_emb, top_k=fetch_k)

        ids = (res.get("ids") or [[]])[0]
        docs = (res.get("documents") or [[]])[0]
        metas = (res.get("metadatas") or [[]])[0]
        dists = (res.get("distances") or [[]])[0]

        qset = {str(x).lower() for x in (query_tags or []) if str(x).strip()}

        hits: List[Dict[str, Any]] = []
        for kid, doc, meta, dist in zip(ids, docs, metas, dists):
            try:
                dist_f = float(dist)
            except Exception:
                dist_f = 1e9
            score = 1.0 - dist_f

            if qset:
                tset = set(_tags_from_meta(meta or {}))
                if tset and (tset & qset):
                    score *= (1.0 + float(tag_boost))

            hits.append({
                "kid": kid,
                "score": float(score),
                "distance": float(dist_f),
                "text": doc or "",
                "meta": meta or {},
            })

        hits.sort(key=lambda x: x["score"], reverse=True)
        return hits[:top_k]
