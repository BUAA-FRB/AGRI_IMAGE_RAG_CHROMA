import os
import json
import hashlib
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Tuple

def _sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8", errors="ignore")).hexdigest()

@dataclass
class KBChunk:
    kid: str
    text: str
    meta: Dict[str, Any]

def _parse_front_matter(raw: str) -> Tuple[Dict[str, Any], str]:
    lines = (raw or "").splitlines()
    if len(lines) < 3 or lines[0].strip() != "---":
        return {}, raw

    meta: Dict[str, Any] = {}
    i = 1
    while i < len(lines):
        if lines[i].strip() == "---":
            return meta, "\n".join(lines[i+1:])
        line = lines[i].strip()
        if ":" in line:
            k, v = line.split(":", 1)
            k, v = k.strip(), v.strip()
            if k == "tags":
                if v.startswith("[") and v.endswith("]"):
                    inner = v[1:-1].strip()
                    meta["tags"] = [x.strip().strip('"').strip("'") for x in inner.split(",") if x.strip()] if inner else []
                else:
                    meta["tags"] = [v]
            else:
                meta[k] = v
        i += 1
    return {}, raw

def _split_by_heading(text: str) -> List[Tuple[str, str]]:
    lines = (text or "").splitlines()
    sections: List[Tuple[str, List[str]]] = []
    cur_title = ""
    cur_lines: List[str] = []
    for ln in lines:
        s = ln.strip()
        if s.startswith("#"):
            if cur_lines:
                sections.append((cur_title, cur_lines))
            cur_title = s.lstrip("#").strip()
            cur_lines = []
        else:
            cur_lines.append(ln)
    if cur_lines:
        sections.append((cur_title, cur_lines))
    out: List[Tuple[str, str]] = []
    for t, ls in sections:
        out.append((t, "\n".join(ls).strip()))
    return out if out else [("", (text or "").strip())]

def _chunk_by_paragraph(text: str, max_chars: int, overlap: int) -> List[str]:
    paras = [p.strip() for p in (text or "").split("\n\n") if p.strip()]
    chunks: List[str] = []
    buf = ""
    for p in paras:
        if not buf:
            buf = p
            continue
        if len(buf) + 2 + len(p) <= max_chars:
            buf = buf + "\n\n" + p
        else:
            chunks.append(buf)
            buf = (buf[-overlap:] + "\n\n" + p).strip() if overlap > 0 else p
    if buf:
        chunks.append(buf)
    return chunks

def load_knowledge(kb_dir: str, max_chars: int = 900, overlap: int = 160) -> Iterable[KBChunk]:
    kb_dir = os.path.abspath(kb_dir)
    docs_dir = os.path.join(kb_dir, "docs")
    candidates: List[str] = []

    if os.path.isdir(docs_dir):
        for root, _, files in os.walk(docs_dir):
            for fn in files:
                if fn.lower().endswith((".md", ".txt")):
                    candidates.append(os.path.join(root, fn))

    jsonl_path = os.path.join(kb_dir, "records.jsonl")
    if os.path.exists(jsonl_path):
        candidates.append(jsonl_path)

    for path in sorted(candidates):
        rel = os.path.relpath(path, kb_dir).replace("\\", "/")

        if path.lower().endswith(".jsonl"):
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                for idx, line in enumerate(f):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except Exception:
                        continue
                    text = str(obj.get("text") or "").strip()
                    if not text:
                        continue
                    tags = obj.get("tags") or []
                    if isinstance(tags, str):
                        tags = [tags]
                    meta = {
                        "path": rel,
                        "title": obj.get("title") or obj.get("id") or rel,
                        "tags": tags,
                        "updated": obj.get("updated") or "",
                        "source": obj.get("source") or "",
                        "record_id": obj.get("id") or f"{rel}#{idx}",
                    }
                    base = obj.get("id") or _sha1(rel + f"#{idx}")[:16]
                    for cidx, chunk in enumerate(_chunk_by_paragraph(text, max_chars, overlap)):
                        yield KBChunk(kid=f"{base}::c{cidx}", text=chunk, meta={**meta, "chunk": cidx})
        else:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                raw = f.read()
            fm, body = _parse_front_matter(raw)
            meta0 = {
                "path": rel,
                "title": fm.get("title") or os.path.splitext(os.path.basename(path))[0],
                "tags": fm.get("tags") or [],
                "updated": fm.get("updated") or "",
                "source": fm.get("source") or "",
            }
            doc_id = _sha1(rel)[:16]
            sections = _split_by_heading(body)
            for si, (sec_title, sec_text) in enumerate(sections):
                if not sec_text.strip():
                    continue
                for cidx, chunk in enumerate(_chunk_by_paragraph(sec_text, max_chars, overlap)):
                    yield KBChunk(
                        kid=f"{doc_id}::s{si}::c{cidx}",
                        text=chunk,
                        meta={**meta0, "section": sec_title, "chunk": cidx},
                    )
