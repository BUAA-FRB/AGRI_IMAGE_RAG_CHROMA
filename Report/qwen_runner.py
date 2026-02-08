from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class QwenGenConfig:
    max_new_tokens: int = 1600
    temperature: float = 0.4
    top_p: float = 0.85
    repetition_penalty: float = 1.05
    do_sample: bool = True
    seed: Optional[int] = 42


class QwenRunner:
    """Lightweight local runner for Qwen2.5-3B-Instruct."""

    def __init__(
        self,
        model_path: str,
        device: str = "auto",
        dtype: str = "auto",
        trust_remote_code: bool = True,
    ) -> None:
        self.model_path = model_path
        self.device = device
        self.dtype = dtype
        self.trust_remote_code = trust_remote_code

        self.tokenizer = None
        self.model = None

    def load(self) -> None:
        if self.tokenizer is not None and self.model is not None:
            return

        # Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=self.trust_remote_code,
            use_fast=False,
        )

        # Model
        torch_dtype = None
        if self.dtype == "auto":
            torch_dtype = "auto"
        else:
            # allow: float16, bfloat16, float32
            torch_dtype = getattr(torch, self.dtype)

        device_map = "auto" if self.device == "auto" else None
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            trust_remote_code=self.trust_remote_code,
            torch_dtype=torch_dtype,
            device_map=device_map,
        )

        if self.device != "auto":
            self.model.to(self.device)

        self.model.eval()

    def generate(self, messages: List[Dict[str, str]], cfg: QwenGenConfig) -> str:
        self.load()

        assert self.tokenizer is not None and self.model is not None

        if cfg.seed is not None:
            torch.manual_seed(int(cfg.seed))

        # Prefer chat_template if available (Qwen Instruct generally supports it)
        if hasattr(self.tokenizer, "apply_chat_template") and getattr(self.tokenizer, "chat_template", None):
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            inputs = self.tokenizer(prompt, return_tensors="pt")
        else:
            # Fallback: naive concatenation
            joined = "\n".join([f"{m['role'].upper()}: {m['content']}" for m in messages]) + "\nASSISTANT:"
            inputs = self.tokenizer(joined, return_tensors="pt")

        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        gen_kwargs: Dict[str, Any] = dict(
            max_new_tokens=int(cfg.max_new_tokens),
            temperature=float(cfg.temperature),
            top_p=float(cfg.top_p),
            repetition_penalty=float(cfg.repetition_penalty),
            do_sample=bool(cfg.do_sample),
            eos_token_id=self.tokenizer.eos_token_id,
        )

        with torch.no_grad():
            out = self.model.generate(**inputs, **gen_kwargs)

        # Remove prompt tokens
        generated = out[0][inputs["input_ids"].shape[1]:]
        text = self.tokenizer.decode(generated, skip_special_tokens=True)

        # Light cleanup
        return text.strip()
