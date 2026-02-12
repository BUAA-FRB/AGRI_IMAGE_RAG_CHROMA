from dataclasses import dataclass
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class Qwen25GenConfig:
    max_new_tokens: int = 512
    temperature: float = 0.2
    top_p: float = 0.9
    do_sample: bool = True


class Qwen25Client:
    def __init__(
        self,
        model_name_or_path: str,
        device_map: str = "auto",
        dtype: str = "auto",
        local_files_only: bool = True,
        trust_remote_code: bool = True,
    ) -> None:
        self.model_name_or_path = model_name_or_path
        self.device_map = device_map
        self.dtype = dtype
        self.local_files_only = local_files_only
        self.trust_remote_code = trust_remote_code

        self._tok = None
        self._model = None

    def _lazy_load(self) -> None:
        if self._model is not None:
            return

        self._tok = AutoTokenizer.from_pretrained(
            self.model_name_or_path,
            local_files_only=self.local_files_only,
            trust_remote_code=self.trust_remote_code,
        )

        if self.dtype == "auto":
            torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        else:
            torch_dtype = getattr(torch, self.dtype)

        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_name_or_path,
            device_map=self.device_map,
            torch_dtype=torch_dtype,
            local_files_only=self.local_files_only,
            trust_remote_code=self.trust_remote_code,
        )
        self._model.eval()

    def generate(self, prompt: str, gen: Optional[Qwen25GenConfig] = None) -> str:
        self._lazy_load()
        assert self._tok is not None and self._model is not None
        gen = gen or Qwen25GenConfig()

        inputs = self._tok(prompt, return_tensors="pt").to(self._model.device)
        with torch.no_grad():
            out_ids = self._model.generate(
                **inputs,
                max_new_tokens=gen.max_new_tokens,
                temperature=gen.temperature,
                top_p=gen.top_p,
                do_sample=gen.do_sample,
            )
        text = self._tok.decode(out_ids[0], skip_special_tokens=True)
        if text.startswith(prompt):
            text = text[len(prompt):].lstrip()
        return text
