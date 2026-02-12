from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import torch


@dataclass
class QwenGenConfig:
    max_new_tokens: int = 1800
    temperature: float = 0.35
    top_p: float = 0.85
    repetition_penalty: float = 1.05
    seed: Optional[int] = 42


class QwenRunner:
    def __init__(self, model_path: str, device: str = "auto", dtype: str = "auto") -> None:
        self.model_path = model_path
        self.device = device
        self.dtype = dtype
        self._tokenizer = None
        self._model = None
        self._resolved_device: Optional[str] = None

    def _pick_device(self) -> str:
        if self.device != "auto":
            return self.device
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def _pick_dtype(self) -> torch.dtype:
        if self.dtype == "float32":
            return torch.float32
        if self.dtype == "float16":
            return torch.float16
        if self.dtype == "bfloat16":
            return torch.bfloat16
        # default
        if torch.cuda.is_available():
            return torch.float16
        return torch.float32

    def _load(self) -> None:
        if self._model is not None and self._tokenizer is not None:
            return

        from transformers import AutoTokenizer, AutoModelForCausalLM

        dev = self._pick_device()
        dt = self._pick_dtype()

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)

        # 一些 Instruct 模型可能没有 pad_token，避免 generate 报 warning/错误
        if self._tokenizer.pad_token_id is None and self._tokenizer.eos_token_id is not None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        # device_map 在 cpu 上不要用
        if dev in ("cuda", "mps"):
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                trust_remote_code=True,
                torch_dtype=dt,
                device_map="auto",
            )
        else:
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                trust_remote_code=True,
                torch_dtype=dt,
                device_map=None,
            )
            self._model = self._model.to(dev)

        self._model.eval()
        self._resolved_device = dev

    @staticmethod
    def _apply_seed(seed: Optional[int]) -> None:
        if seed is None:
            return
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def _build_prompt_text(self, messages: List[Dict[str, str]]) -> str:
        """
        最稳策略：先拿到纯文本 prompt，再做 tokenizer。
        避免 apply_chat_template(tokenize=True, return_tensors="pt") 在不同版本下返回类型不一致。
        """
        assert self._tokenizer is not None
        tok = self._tokenizer

        # tokenize=False => 返回字符串（稳定）
        prompt_text = tok.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        if not isinstance(prompt_text, str):
            prompt_text = str(prompt_text)
        return prompt_text

    def generate(self, messages: List[Dict[str, str]], cfg: QwenGenConfig) -> str:
        self._load()
        assert self._tokenizer is not None and self._model is not None

        self._apply_seed(cfg.seed)

        tok = self._tokenizer
        model = self._model

        # 1) messages -> prompt text
        prompt_text = self._build_prompt_text(messages)

        # 2) prompt text -> tensors
        enc = tok(
            prompt_text,
            return_tensors="pt",
            padding=False,
            truncation=True,
        )

        # 3) move to model device
        enc = {k: v.to(model.device) for k, v in enc.items()}

        pad_id = tok.pad_token_id
        if pad_id is None:
            pad_id = tok.eos_token_id

        eos_id = tok.eos_token_id

        with torch.no_grad():
            out_ids = model.generate(
                **enc,  # ✅ 关键修复：解包 BatchEncoding
                max_new_tokens=int(cfg.max_new_tokens),
                do_sample=True,
                temperature=float(cfg.temperature),
                top_p=float(cfg.top_p),
                repetition_penalty=float(cfg.repetition_penalty),
                pad_token_id=pad_id,
                eos_token_id=eos_id,
            )

        # 4) 只解码新增部分（避免把 prompt 重复输出）
        input_len = enc["input_ids"].shape[-1]
        gen_ids = out_ids[0][input_len:]
        text = tok.decode(gen_ids, skip_special_tokens=True)

        # 你原来的 heuristic：保留最后一次出现的 "### 3.1" 之后的内容
        # 但注意：这里 text 已经是“新生成部分”，一般不需要再剪；保留逻辑以兼容你下游解析
        pos = text.rfind("### 3.1")
        if pos != -1:
            text = text[pos:]

        return (text or "").strip()
