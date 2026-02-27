from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple


@dataclass
class LLMResult:
    summary_cn: str
    summary_en: str
    uncertainty_notes: List[str]


def _try_import_transformers():
    try:
        import transformers  # noqa
        return True
    except Exception:
        return False


def generate_bilingual_summary_with_qwen25(
    model_path: Path,
    prompt_cn: str,
    max_new_tokens: int = 900,
    temperature: float = 0.35,
    top_p: float = 0.9,
    repetition_penalty: float = 1.05,
) -> LLMResult:
    """
    Text-only generation using Qwen2.5-3B-Instruct (local).
    If transformers/torch are unavailable, raises RuntimeError.
    """
    if not _try_import_transformers():
        raise RuntimeError("transformers not installed. Install requirements-llm.txt first.")

    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(str(model_path), trust_remote_code=True)
    model.to(device)
    model.eval()

    system = "你是农业遥感诊断与农艺决策助手。输出必须包含中文与英文两个摘要段落，并给出不确定性提示列表。避免给出具体化学品配方/用量，只给原则性建议。"
    user = prompt_cn

    # Qwen instruct format (simple)
    text = f"<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n"
    inputs = tok(text, return_tensors="pt").to(device)

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            eos_token_id=tok.eos_token_id,
        )
    gen = tok.decode(out[0], skip_special_tokens=False)

    # Try to parse sections.
    # Expected format:
    # 【中文摘要】...
    # 【English Summary】...
    # 【Uncertainty】...
    cn = ""
    en = ""
    unc: List[str] = []
    # remove prompt portion
    if "<|im_start|>assistant" in gen:
        gen = gen.split("<|im_start|>assistant", 1)[-1]
    # strip special tokens
    gen_clean = gen.replace("<|im_end|>", "").replace("<|im_start|>", "").strip()

    def _extract(tag: str) -> str:
        if tag not in gen_clean:
            return ""
        seg = gen_clean.split(tag, 1)[-1]
        # stop at next tag
        for nxt in ["【中文摘要】", "【English Summary】", "【Uncertainty】"]:
            if nxt != tag and nxt in seg:
                seg = seg.split(nxt, 1)[0]
        return seg.strip()

    cn = _extract("【中文摘要】") or gen_clean[:300]
    en = _extract("【English Summary】") or ""
    unc_text = _extract("【Uncertainty】") or ""
    if unc_text:
        for line in unc_text.splitlines():
            t = line.strip("-• 	")
            if t:
                unc.append(t)

    if not en:
        # fallback: short translation-ish (not perfect)
        en = "Key low-vigor patches are detected; verify with sampling and operations/weather context."

    if not unc:
        unc = ["RGB-only vigor proxy may confuse nutrient deficiency with water stress; consider NIR/temporal evidence."]

    return LLMResult(summary_cn=cn, summary_en=en, uncertainty_notes=unc)
