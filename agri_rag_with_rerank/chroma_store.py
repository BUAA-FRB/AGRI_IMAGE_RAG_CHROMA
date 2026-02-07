# agri_rag/chroma_store.py
from __future__ import annotations
from typing import Any, Dict, List, Optional
import chromadb
from chromadb.config import Settings
from .types import RetrievalHit
import io
import logging
from contextlib import contextmanager, redirect_stdout, redirect_stderr

logger = logging.getLogger(__name__)


def get_client(db_dir: str) -> chromadb.PersistentClient:
    return chromadb.PersistentClient(
        path=db_dir,
        settings=Settings(anonymized_telemetry=False),
    )


def get_or_create_collection(db_dir: str, name: str) -> chromadb.api.models.Collection.Collection:
    client = get_client(db_dir)
    col = client.get_or_create_collection(
        name=name,
        metadata={"hnsw:space": "cosine"},
    )
    logger.debug("Chroma collection ready: db_dir=%s name=%s", db_dir, name)
    return col


@contextmanager
def _suppress_stdout_stderr(log_captured: bool = True):
    """
    Suppress noisy prints/logs from some Chroma versions.
    If log_captured=True, captured content will be logged at DEBUG.
    """
    buf_out = io.StringIO()
    buf_err = io.StringIO()
    with redirect_stdout(buf_out), redirect_stderr(buf_err):
        yield
    if log_captured:
        o = buf_out.getvalue().strip()
        e = buf_err.getvalue().strip()
        if o:
            logger.debug("Captured stdout from Chroma:\n%s", o[:4000])
        if e:
            logger.debug("Captured stderr from Chroma:\n%s", e[:4000])


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
    Add with batch logging (no tqdm terminal output).
    - Prefer collection.upsert (if available)
    - Fallback to delete+add, but suppress noisy delete output.
    """
    n = len(ids)
    if n == 0:
        logger.info("Chroma upsert skipped: empty ids")
        return

    batch_size = 256
    col_name = getattr(collection, "name", "collection")

    logger.info("Chroma write start: col=%s items=%d batch_size=%d", col_name, n, batch_size)

    if hasattr(collection, "upsert"):
        done = 0
        for s, e in _batch_slices(n, batch_size):
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
            done = e
            if done % (batch_size * 10) == 0 or done == n:
                logger.info("Chroma upsert progress: col=%s %d/%d", col_name, done, n)

        logger.info("Chroma upsert done: col=%s items=%d", col_name, n)
        return

    # Fallback path: delete + add
    try:
        with _suppress_stdout_stderr(log_captured=True):
            collection.delete(ids=ids)
    except Exception as ex:
        logger.debug("Chroma delete ignored: %s", ex)

    done = 0
    for s, e in _batch_slices(n, batch_size):
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
        done = e
        if done % (batch_size * 10) == 0 or done == n:
            logger.info("Chroma add progress: col=%s %d/%d", col_name, done, n)

    logger.info("Chroma add done: col=%s items=%d", col_name, n)


def query(
    collection,
    query_embedding: List[float],
    top_k: int = 5,
) -> List[RetrievalHit]:
    logger.debug("Chroma query: col=%s top_k=%d", getattr(collection, "name", "collection"), top_k)

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

    logger.debug("Chroma query done: hits=%d", len(hits))
    return hits
