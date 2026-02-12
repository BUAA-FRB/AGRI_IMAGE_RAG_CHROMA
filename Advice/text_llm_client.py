# Advice/text_llm_client.py
import os
from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class GenConfig:
    max_new_tokens: int = 900
    temperature: float = 0.2
    top_p: float = 0.9
    do_sample: bool = True


def _looks_like_local_path(s: str) -> bool:
    if not s:
        return False
    x = s.strip()
    # Windows: C:\..., .\..., ..\..., 包含斜杠/反斜杠基本都可认为是路径
    if x.startswith(".\\") or x.startswith("./") or x.startswith("..\\") or x.startswith("../"):
        return True
    if ":\\" in x or ":/" in x:
        return True
    if "\\" in x or "/" in x:
        return True
    return False


class Qwen25TextClient:
    """
    本地 Qwen2.5-3B-Instruct 文本生成：
    - 如果传入看起来是“本地路径”，但目录不存在：直接 FileNotFoundError
      （避免 transformers 把它当 HF repo_id 触发校验错误）
    - 本地目录自动开启离线变量（更稳）
    """

    def __init__(
        self,
        model_path: str,
        device_map: str = "auto",
        local_files_only: bool = True,
        trust_remote_code: bool = True,
    ):
        if not model_path:
            raise ValueError("model_path 不能为空（请传入本地 Qwen2.5-3B-Instruct 目录）")

        # 统一绝对路径（用户传 .\\models\\... 也没问题）
        mp_abs = os.path.abspath(model_path)

        # 如果像路径，就必须存在目录；不存在就直接报错（不走 repo_id）
        if _looks_like_local_path(model_path) and not os.path.isdir(mp_abs):
            raise FileNotFoundError(
                f"本地模型目录不存在：{mp_abs}\n"
                "请确认你传入的是本地目录，例如：.\\models\\Qwen\\Qwen2.5-3B-Instruct"
            )

        self.model_path = mp_abs if os.path.isdir(mp_abs) else model_path  # 允许 repo_id
        self.device_map = device_map
        self.local_files_only = local_files_only
        self.trust_remote_code = trust_remote_code
        self._tok = None
        self._model = None

        # 若是本地目录且要求离线：设置离线环境变量
        if os.path.isdir(mp_abs) and self.local_files_only:
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
            os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    def _lazy_load(self):
        if self._tok is not None and self._model is not None:
            return

        from transformers import AutoTokenizer, AutoModelForCausalLM  # type: ignore

        self._tok = AutoTokenizer.from_pretrained(
            self.model_path,
            local_files_only=self.local_files_only,
            trust_remote_code=self.trust_remote_code,
        )

        kwargs = dict(
            local_files_only=self.local_files_only,
            trust_remote_code=self.trust_remote_code,
            device_map=self.device_map,
            torch_dtype="auto",
            low_cpu_mem_usage=True,
        )

        # 尽量走 safetensors（避免 torch.load 版本限制）
        try:
            kwargs["use_safetensors"] = True
            self._model = AutoModelForCausalLM.from_pretrained(self.model_path, **kwargs)
        except TypeError:
            kwargs.pop("use_safetensors", None)
            self._model = AutoModelForCausalLM.from_pretrained(self.model_path, **kwargs)

        self._model.eval()

        # pad_token 兜底
        if getattr(self._tok, "pad_token_id", None) is None and getattr(self._tok, "eos_token_id", None) is not None:
            self._tok.pad_token_id = self._tok.eos_token_id

    def chat(self, messages: List[Dict[str, str]], gen: Optional[GenConfig] = None) -> str:
        self._lazy_load()
        gen = gen or GenConfig()

        if hasattr(self._tok, "apply_chat_template"):
            prompt = self._tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            prompt = ""
            for m in messages:
                prompt += f"[{m['role'].upper()}]\n{m['content']}\n"
            prompt += "[ASSISTANT]\n"

        import torch

        enc = self._tok(prompt, return_tensors="pt")
        input_ids = enc["input_ids"]
        attn = enc.get("attention_mask", None)

        # device_map=auto：输入放 CPU 也可以，transformers 会处理分发
        with torch.inference_mode():
            out_ids = self._model.generate(
                input_ids=input_ids,
                attention_mask=attn,
                max_new_tokens=gen.max_new_tokens,
                temperature=gen.temperature,
                top_p=gen.top_p,
                do_sample=gen.do_sample,
                pad_token_id=getattr(self._tok, "pad_token_id", None),
                eos_token_id=getattr(self._tok, "eos_token_id", None),
            )

        # token 级别切掉 prompt（比字符串切割稳定）
        gen_part = out_ids[0, input_ids.shape[1]:]
        text = self._tok.decode(gen_part, skip_special_tokens=True)
        return text.strip()
