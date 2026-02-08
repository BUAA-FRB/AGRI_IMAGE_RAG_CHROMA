# Advice/chroma_store.py
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import chromadb
from chromadb.config import Settings


def _client(db_dir: str) -> chromadb.PersistentClient:
    os.makedirs(db_dir, exist_ok=True)
    return chromadb.PersistentClient(
        path=db_dir,
        settings=Settings(anonymized_telemetry=False),
    )


def _normalize_metric(metric: Optional[str]) -> Optional[str]:
    if metric is None:
        return None
    m = str(metric).lower().strip()
    if m in ("cos", "cosine"):
        return "cosine"
    if m in ("l2", "euclidean"):
        return "l2"
    if m in ("ip", "inner_product", "dot"):
        return "ip"
    return m


def get_or_create_collection(
    db_dir: str,
    name: str,
    metadata: Optional[Dict[str, Any]] = None,
    metric: Optional[str] = None,
):
    client = _client(db_dir)

    md = dict(metadata or {})
    m = _normalize_metric(metric)
    if m:
        md.setdefault("hnsw:space", m)

    try:
        return client.get_collection(name)
    except Exception:
        return client.create_collection(name, metadata=md)


def reset_collection(db_dir: str, name: str, metric: Optional[str] = None) -> None:
    client = _client(db_dir)
    try:
        client.delete_collection(name)
    except Exception:
        pass

    md: Dict[str, Any] = {}
    m = _normalize_metric(metric)
    if m:
        md["hnsw:space"] = m
    client.create_collection(name, metadata=md)


def upsert(
    collection,
    ids: List[str],
    embeddings: List[List[float]],
    metadatas: Optional[List[Dict[str, Any]]] = None,
    documents: Optional[List[str]] = None,
) -> None:
    if not ids:
        return
    collection.upsert(
        ids=ids,
        embeddings=embeddings,
        metadatas=metadatas,
        documents=documents,
    )


def query(
    collection,
    embedding: List[float],
    top_k: int = 8,
    where: Optional[Dict[str, Any]] = None,
    include_embeddings: bool = False,
):
    include = ["documents", "metadatas", "distances"]
    if include_embeddings:
        include.append("embeddings")

    return collection.query(
        query_embeddings=[embedding],
        n_results=top_k,
        where=where,
        include=include,
    )
