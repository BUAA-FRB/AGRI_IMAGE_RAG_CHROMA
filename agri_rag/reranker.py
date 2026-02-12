# agri_rag/reranker.py
from __future__ import annotations

import os
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .types import RetrievalHit
from .prompt import build_qwen_rerank_messages
from .structured_io import extract_json_object

logger = logging.getLogger(__name__)


@dataclass
class QwenTextGenConfig:
    max_new_tokens: int = 768
    temperature: float = 0.0
    top_p: float = 0.9
    do_sample: Optional[bool] = None


def _is_local_dir(p: str) -> bool:
    try:
        return Path(p).exists() and Path(p).is_dir()
    except Exception:
        return False


def _repo_id_to_local_dir(local_root: str, repo_id: str) -> str:
    return str(Path(local_root) / Path(repo_id.replace("/", os.sep)))


def ensure_hf_snapshot_local(
    repo_id: str,
    local_root: str = "./models",
    revision: Optional[str] = None,
    force_download: bool = False,
    token: Optional[str] = None,
) -> str:
    local_root_path = Path(local_root).resolve()
    local_root_path.mkdir(parents=True, exist_ok=True)

    local_dir = Path(_repo_id_to_local_dir(str(local_root_path), repo_id))
    local_dir.mkdir(parents=True, exist_ok=True)

    cache_dir = local_root_path / "_hf_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    if not force_download:
        if (local_dir / "config.json").exists() or (local_dir / "model.safetensors.index.json").exists() or any(local_dir.glob("*.safetensors")):
            logger.info("Reranker snapshot exists: %s", local_dir)
            return str(local_dir)

    try:
        from huggingface_hub import snapshot_download

        logger.info("Downloading reranker snapshot: repo_id=%s -> %s", repo_id, local_dir)
        snapshot_download(
            repo_id=repo_id,
            revision=revision,
            local_dir=str(local_dir),
            local_dir_use_symlinks=False,
            cache_dir=str(cache_dir),
            resume_download=True,
            force_download=force_download,
            token=token,
        )
        return str(local_dir)
    except Exception as e:
        if (local_dir / "config.json").exists():
            logger.warning("Download failed but config exists, will try loading from: %s err=%s", local_dir, e)
            return str(local_dir)
        raise RuntimeError(
            f"Failed to download text model repo '{repo_id}' into '{local_dir}'. "
            f"Check network / token / repo id. Original error: {e}"
        ) from e


def _messages_to_plain_text(messages: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    for m in messages:
        role = str(m.get("role", "user"))
        content = m.get("content", "")

        if isinstance(content, str):
            out.append({"role": role, "content": content})
            continue

        if isinstance(content, list):
            parts: List[str] = []
            for it in content:
                if not isinstance(it, dict):
                    continue
                if it.get("type") == "text" and isinstance(it.get("text"), str):
                    parts.append(it["text"])
            out.append({"role": role, "content": "\n".join(parts).strip()})
            continue

        out.append({"role": role, "content": str(content)})

    return out


class QwenTextClient:
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
        local_root: str = "./models",
        revision: Optional[str] = None,
        force_download: bool = False,
        hf_token: Optional[str] = None,
        device_map: str = "auto",
        dtype: str = "auto",
        attn_implementation: Optional[str] = None,
        trust_remote_code: bool = False,
    ) -> None:
        self.model_name = model_name
        self.local_root = local_root
        self.revision = revision
        self.force_download = force_download
        self.hf_token = hf_token
        self.device_map = device_map
        self.dtype = dtype
        self.attn_implementation = attn_implementation
        self.trust_remote_code = trust_remote_code

        self._resolved_local_path: Optional[str] = None
        self._model = None
        self._tokenizer = None

    def _resolve_local_path(self) -> str:
        if self._resolved_local_path is not None:
            return self._resolved_local_path

        if _is_local_dir(self.model_name):
            self._resolved_local_path = str(Path(self.model_name).resolve())
            return self._resolved_local_path

        local_dir = ensure_hf_snapshot_local(
            repo_id=self.model_name,
            local_root=self.local_root,
            revision=self.revision,
            force_download=self.force_download,
            token=self.hf_token,
        )
        self._resolved_local_path = str(Path(local_dir).resolve())
        return self._resolved_local_path

    @property
    def resolved_local_path(self) -> str:
        return self._resolve_local_path()

    def _lazy_load(self) -> None:
        if self._model is not None and self._tokenizer is not None:
            return

        local_path = self._resolve_local_path()
        logger.info("Loading reranker model from: %s", local_path)

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tok = AutoTokenizer.from_pretrained(
            local_path,
            local_files_only=True,
            trust_remote_code=self.trust_remote_code,
        )

        model_kwargs: Dict[str, Any] = {
            "device_map": self.device_map,
            "local_files_only": True,
            "trust_remote_code": self.trust_remote_code,
            "low_cpu_mem_usage": True,
        }

        if self.attn_implementation is not None:
            model_kwargs["attn_implementation"] = self.attn_implementation

        if self.dtype == "auto":
            model_kwargs["torch_dtype"] = "auto"
        else:
            if not hasattr(torch, self.dtype):
                raise ValueError(f"Unknown dtype={self.dtype}, use e.g. 'float16','bfloat16','auto'")
            model_kwargs["torch_dtype"] = getattr(torch, self.dtype)

        model = AutoModelForCausalLM.from_pretrained(local_path, **model_kwargs)
        model.eval()

        self._tokenizer = tok
        self._model = model
        logger.info("Reranker ready: %s", local_path)

    def chat(self, messages: List[Dict[str, Any]], gen: Optional[QwenTextGenConfig] = None) -> str:
        self._lazy_load()
        assert self._model is not None and self._tokenizer is not None

        if gen is None:
            gen = QwenTextGenConfig()

        import torch

        msg_text = _messages_to_plain_text(messages)

        if hasattr(self._tokenizer, "apply_chat_template"):
            prompt = self._tokenizer.apply_chat_template(
                msg_text,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            prompt_lines = []
            for m in msg_text:
                prompt_lines.append(f"{m['role']}: {m['content']}")
            prompt_lines.append("assistant:")
            prompt = "\n".join(prompt_lines)

        inputs = self._tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self._model.device) for k, v in inputs.items()}

        do_sample = gen.do_sample
        if do_sample is None:
            do_sample = gen.temperature is not None and float(gen.temperature) > 0.0

        generate_kwargs: Dict[str, Any] = {
            "max_new_tokens": int(gen.max_new_tokens),
            "do_sample": bool(do_sample),
        }
        if do_sample:
            generate_kwargs["temperature"] = float(gen.temperature)
            generate_kwargs["top_p"] = float(gen.top_p)

        with torch.inference_mode():
            out_ids = self._model.generate(**inputs, **generate_kwargs)

        in_len = inputs["input_ids"].shape[1]
        gen_ids = out_ids[0][in_len:]
        text = self._tokenizer.decode(gen_ids, skip_special_tokens=True)
        return text.strip()


def _clamp01(x: float) -> float:
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return x


def _distance_fallback_scores(cand: List[RetrievalHit]) -> List[float]:
    ds = [float(h.distance) for h in cand]
    if not ds:
        return []
    dmin = min(ds)
    dmax = max(ds)
    denom = max(1e-9, dmax - dmin)
    sims = [1.0 - (d - dmin) / denom for d in ds]
    return [_clamp01(s) for s in sims]


def _parse_llm_ranking(obj: Dict[str, Any], top_n: int) -> Tuple[Dict[int, float], Dict[int, str]]:
    scored: Dict[int, float] = {}
    reasons: Dict[int, str] = {}

    if not isinstance(obj, dict):
        return scored, reasons

    ranking = obj.get("ranking", None)

    if ranking is None and isinstance(obj.get("scores", None), dict):
        d = obj["scores"]
        for k, v in d.items():
            if not (isinstance(k, str) and k.startswith("E")):
                continue
            try:
                idx = int(k[1:]) - 1
            except Exception:
                continue
            if 0 <= idx < top_n:
                try:
                    s = float(v)
                except Exception:
                    continue
                scored[idx] = _clamp01(s)
        return scored, reasons

    if not isinstance(ranking, list):
        return scored, reasons

    for it in ranking:
        if not isinstance(it, dict):
            continue
        eid = it.get("eid", None)
        score = it.get("score", None)
        reason = it.get("reason", "")
        if not (isinstance(eid, str) and eid.startswith("E")):
            continue
        try:
            idx = int(eid[1:]) - 1
        except Exception:
            continue
        if 0 <= idx < top_n:
            try:
                s = float(score)
            except Exception:
                continue
            scored[idx] = _clamp01(s)
            if isinstance(reason, str) and reason.strip():
                reasons[idx] = reason.strip()

    return scored, reasons


def _build_repair_messages(user_query: str, cand: List[RetrievalHit], top_n: int, prev_out: str) -> List[Dict[str, Any]]:
    eids = [f"E{i}" for i in range(1, top_n + 1)]
    lines: List[str] = []
    lines.append("你上一次的输出未满足程序可解析要求（可能缺少 eid 或 JSON 不完整）。")
    lines.append("请严格按要求修复输出。")
    lines.append("")
    lines.append("【硬性要求】")
    lines.append("1) 只输出一个 JSON 对象，不要输出任何额外文字/解释/Markdown。")
    lines.append(f"2) ranking 必须包含恰好 {top_n} 条，必须覆盖全部 eid：{', '.join(eids)}。")
    lines.append("3) score 必须是 0~1 的数字，不能为 null。")
    lines.append("4) score 表示语义相关性（基于 labels_present 等元信息），不要把相似度提示当作 score。")
    lines.append("")
    lines.append(f"用户问题：{user_query}")
    lines.append("")
    lines.append("候选证据：")
    for i, h in enumerate(cand, 1):
        m = h.metadata or {}
        labels_present = m.get("labels_present", "")
        tile_id = m.get("tile_id", "")
        split = m.get("split", "")
        lines.append(f"- E{i}: sim_rank={i}/{top_n}, tile_id={tile_id}, split={split}, labels_present={labels_present}")
    lines.append("")
    lines.append("你上一次的输出（仅供参考，不要复述）：")
    lines.append(prev_out.strip()[:1200])
    lines.append("")
    lines.append("【输出 JSON schema】")
    lines.append('{ "ranking": [ {"eid":"E1","score":0.0,"reason":"..."} ] }')

    content = [{"type": "text", "text": "\n".join(lines)}]
    return [{"role": "user", "content": content}]


def rerank_hits_with_qwen(
    user_query: str,
    hits: List[RetrievalHit],
    client: QwenTextClient,
    top_n: int = 30,
    gen: Optional[QwenTextGenConfig] = None,
) -> Tuple[List[RetrievalHit], Dict[str, Any]]:
    top_n = max(1, min(top_n, len(hits)))
    cand = hits[:top_n]

    messages = build_qwen_rerank_messages(user_query=user_query, hits=cand, max_hits=top_n)
    out1 = client.chat(messages, gen=gen)
    obj1 = extract_json_object(out1) or {}
    scored1, reasons1 = _parse_llm_ranking(obj1, top_n=top_n)

    coverage1 = len(scored1) / float(top_n) if top_n > 0 else 0.0
    out2 = ""
    obj2: Dict[str, Any] = {}
    scored2: Dict[int, float] = {}
    reasons2: Dict[int, str] = {}
    used_repair = False

    if coverage1 < 0.9:
        used_repair = True
        repair_messages = _build_repair_messages(user_query=user_query, cand=cand, top_n=top_n, prev_out=out1)
        out2 = client.chat(repair_messages, gen=gen)
        obj2 = extract_json_object(out2) or {}
        scored2, reasons2 = _parse_llm_ranking(obj2, top_n=top_n)

    scored_llm = scored2 if (used_repair and len(scored2) >= len(scored1)) else scored1
    reasons_llm = reasons2 if (used_repair and len(scored2) >= len(scored1)) else reasons1
    parsed_obj = obj2 if (used_repair and len(scored2) >= len(scored1)) else obj1
    raw_output = out2 if (used_repair and len(scored2) >= len(scored1)) else out1

    coverage = len(scored_llm) / float(top_n) if top_n > 0 else 0.0

    fallback_scores = _distance_fallback_scores(cand)
    if len(fallback_scores) != top_n:
        fallback_scores = [0.0 for _ in range(top_n)]

    semantic_scores = fallback_scores[:]
    for i, s in scored_llm.items():
        semantic_scores[i] = _clamp01(float(s))

    vals = [semantic_scores[i] for i in range(top_n)]
    score_range = (max(vals) - min(vals)) if vals else 0.0
    has_semantic_signal = (coverage >= 0.6) and (score_range >= 1e-3)

    used_fallback = coverage < 1.0

    if has_semantic_signal:
        final_scores = [_clamp01(0.92 * semantic_scores[i] + 0.08 * fallback_scores[i]) for i in range(top_n)]
        used_position_score = False
    else:
        final_scores = fallback_scores[:]
        used_fallback = True
        used_position_score = True

    before_order = [f"E{i+1}" for i in range(top_n)]
    order = list(range(top_n))
    order.sort(key=lambda i: (final_scores[i], -float(cand[i].distance), -i), reverse=True)
    after_order = [f"E{i+1}" for i in order]

    reranked = [cand[i] for i in order]
    reranked.extend(hits[top_n:])

    effective = (after_order != before_order)

    debug = {
        "raw_output": raw_output,
        "parsed": parsed_obj,
        "used_repair": used_repair,
        "repair_debug": {
            "coverage_before": coverage1,
            "coverage_after": (len(scored2) / float(top_n) if (used_repair and top_n > 0) else None),
            "used_output": "repair" if (used_repair and len(scored2) >= len(scored1)) else "first",
            "first_raw_output": out1 if used_repair else None,
            "repair_raw_output": out2 if used_repair else None,
        } if used_repair else {},
        "effective": effective,
        "has_semantic_signal": has_semantic_signal,
        "coverage": coverage,
        "score_range": float(score_range),
        "used_fallback": used_fallback,
        "used_position_score": used_position_score,
        "scores": {f"E{i+1}": float(final_scores[i]) for i in range(top_n)},
        "semantic_scores": {f"E{i+1}": float(semantic_scores[i]) for i in range(top_n)},
        "fallback_scores": {f"E{i+1}": float(fallback_scores[i]) for i in range(top_n)},
        "reasons": {f"E{i+1}": reasons_llm.get(i, "") for i in range(top_n)},
        "top_n": top_n,
        "model_local_path": getattr(client, "resolved_local_path", None),
        "model_name": getattr(client, "model_name", None),
        "before_order": before_order,
        "after_order": after_order,
    }

    logger.info("Rerank done: top_n=%d effective=%s coverage=%.2f semantic_signal=%s",
                top_n, effective, coverage, has_semantic_signal)
    return reranked, debug
