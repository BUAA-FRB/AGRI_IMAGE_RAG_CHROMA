from __future__ import annotations
from typing import Any, Dict, List, Optional
import chromadb
from chromadb.config import Settings
from .types import RetrievalHit
from tqdm import tqdm
import io
from contextlib import contextmanager, redirect_stdout, redirect_stderr

def get_client(db_dir: str) -> chromadb.PersistentClient:
    return chromadb.PersistentClient(
        path=db_dir,
        settings=Settings(anonymized_telemetry=False),
    )

def get_or_create_collection(db_dir: str, name: str) -> chromadb.api.models.Collection.Collection:
    client = get_client(db_dir)
    return client.get_or_create_collection(
        name=name,
        metadata={"hnsw:space": "cosine"},
    )

@contextmanager
def _suppress_stdout_stderr():
    """
    Suppress noisy prints/logs from some Chroma versions (e.g. delete nonexisting ids).
    This does NOT affect your data; it only hides console output in this block.
    """
    buf_out = io.StringIO()
    buf_err = io.StringIO()
    with redirect_stdout(buf_out), redirect_stderr(buf_err):
        yield

def _batch_slices(n: int, batch_size: int):
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        yield start, end

def upsert(
    collection,
    ids: List[str],
    embeddings: List[List[float]],
    metadatas: List[Dict[str, Any]],
    documents: Optional[List[str]] = None,
) -> None:
    """
    Add with progress bar.
    - Prefer collection.upsert (if available) to avoid delete noise and duplicate-id issues.
    - Fallback to delete+add, but suppress noisy delete output.
    """
    n = len(ids)
    if n == 0:
        return
  
    batch_size = 256

    col_name = getattr(collection, "name", "collection")
    if hasattr(collection, "upsert"):
        for s, e in tqdm(list(_batch_slices(n, batch_size)), desc=f"Chroma upsert ({col_name})"):
            if documents is None:
                collection.upsert(
                    ids=ids[s:e],
                    embeddings=embeddings[s:e],
                    metadatas=metadatas[s:e],
                )
            else:
                collection.upsert(
                    ids=ids[s:e],
                    embeddings=embeddings[s:e],
                    metadatas=metadatas[s:e],
                    documents=documents[s:e],
                )
        return
    try:
        with _suppress_stdout_stderr():
            collection.delete(ids=ids)
    except Exception:
        pass

    for s, e in tqdm(list(_batch_slices(n, batch_size)), desc=f"Chroma add ({col_name})"):
        if documents is None:
            collection.add(
                ids=ids[s:e],
                embeddings=embeddings[s:e],
                metadatas=metadatas[s:e],
            )
        else:
            collection.add(
                ids=ids[s:e],
                embeddings=embeddings[s:e],
                metadatas=metadatas[s:e],
                documents=documents[s:e],
            )


def query(
    collection,
    query_embedding: List[float],
    top_k: int = 5,
) -> List[RetrievalHit]:
    res = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["metadatas", "documents", "distances"],
    )
    ids = res.get("ids", [[]])[0]
    metas = res.get("metadatas", [[]])[0]
    docs = res.get("documents", [[]])[0]
    dists = res.get("distances", [[]])[0]

    hits: List[RetrievalHit] = []
    for _id, m, d, doc in zip(ids, metas, dists, docs):
        hits.append(RetrievalHit(id=_id, distance=float(d), metadata=m or {}, document=doc))
    return hits
