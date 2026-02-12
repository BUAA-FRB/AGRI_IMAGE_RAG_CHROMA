from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional


@dataclass
class QwenGenConfig:
    max_new_tokens: int = 1200
    temperature: float = 0.4
    top_p: float = 0.9
    repetition_penalty: float = 1.05


class LocalQwenClient:
    """
    Local Qwen inference via HuggingFace Transformers.

    Model dir default expected:
      ./models/Qwen/Qwe2.5-3B-Instruct
    (also tries ./models/Qwen/Qwen2.5-3B-Instruct)

    Notes:
    - Requires torch + transformers + accelerate (recommended).
    - device_map="auto" will try GPU first if available.
    """

    def __init__(self, model_dir: str | Path) -> None:
        self.model_dir = Path(model_dir)
        self._tokenizer = None
        self._model = None

    def _resolve_dir(self) -> Path:
        if self.model_dir.exists():
            return self.model_dir
        # fallback: common typo correction
        alt = self.model_dir.parent / "Qwen2.5-3B-Instruct"
        if alt.exists():
            return alt
        raise FileNotFoundError(
            f"Local Qwen model directory not found: {self.model_dir} (also tried {alt})."
        )

    def load(self) -> None:
        if self._model is not None and self._tokenizer is not None:
            return

        model_dir = self._resolve_dir()

        try:
            import torch  # noqa
            from transformers import AutoTokenizer, AutoModelForCausalLM  # noqa
        except Exception as e:
            raise RuntimeError(
                "Missing dependencies for local Qwen inference. "
                "Please install: torch transformers accelerate sentencepiece"
            ) from e

        self._tokenizer = AutoTokenizer.from_pretrained(
            str(model_dir),
            trust_remote_code=True,
            use_fast=True,
        )
        self._model = AutoModelForCausalLM.from_pretrained(
            str(model_dir),
            torch_dtype="auto",
            device_map="auto",
            trust_remote_code=True,
        )
        self._model.eval()

    def generate_markdown(self, prompt: Dict[str, str], gen_cfg: Optional[QwenGenConfig] = None) -> str:
        self.load()
        gen_cfg = gen_cfg or QwenGenConfig()

        tokenizer = self._tokenizer
        model = self._model
        assert tokenizer is not None and model is not None

        try:
            import torch
        except Exception:
            torch = None  # type: ignore

        messages = [
            {"role": "system", "content": prompt["system"]},
            {"role": "user", "content": prompt["user"]},
        ]

        # chat template if available
        if hasattr(tokenizer, "apply_chat_template"):
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(text, return_tensors="pt")
        else:
            # fallback: simple concatenation
            text = f"[SYSTEM]\n{prompt['system']}\n\n[USER]\n{prompt['user']}\n\n[ASSISTANT]\n"
            inputs = tokenizer(text, return_tensors="pt")

        if hasattr(model, "device"):
            inputs = {k: v.to(model.device) for k, v in inputs.items()}

        gen_kwargs = dict(
            max_new_tokens=int(gen_cfg.max_new_tokens),
            temperature=float(gen_cfg.temperature),
            top_p=float(gen_cfg.top_p),
            repetition_penalty=float(gen_cfg.repetition_penalty),
            do_sample=True if gen_cfg.temperature > 0 else False,
        )

        if torch is not None:
            with torch.inference_mode():
                out = model.generate(**inputs, **gen_kwargs)
        else:
            out = model.generate(**inputs, **gen_kwargs)

        # decode only the generated part
        output_text = tokenizer.decode(out[0], skip_special_tokens=True)

        # Heuristic: try to cut away the prompt if echoed
        # Keep last occurrence of assistant marker-like tokens
        # (Not perfect, but works for many instruct models)
        for marker in ["[ASSISTANT]", "assistant\n", "Assistant:\n"]:
            if marker in output_text:
                output_text = output_text.split(marker)[-1].strip()

        return output_text.strip()
