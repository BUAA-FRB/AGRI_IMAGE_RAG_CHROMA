# agri_rag/qwen3_vl_client.py
from __future__ import annotations

import os
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class Qwen3VLGenConfig:
    max_new_tokens: int = 512
    temperature: float = 0.2
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
    local_root: str = "./model",
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
        if (local_dir / "config.json").exists() or any(local_dir.glob("*.safetensors")) or (local_dir / "model.safetensors.index.json").exists():
            logger.info("Qwen3-VL snapshot already exists: %s", local_dir)
            return str(local_dir)

    try:
        from huggingface_hub import snapshot_download

        logger.info("Downloading Qwen3-VL snapshot: repo_id=%s -> %s", repo_id, local_dir)
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
            f"Failed to download model repo '{repo_id}' into '{local_dir}'. "
            f"Please check network / token / repo id. Original error: {e}"
        ) from e


class Qwen3VLClient:
    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-VL-7B-Instruct",
        local_root: str = "./model",
        revision: Optional[str] = None,
        force_download: bool = False,
        hf_token: Optional[str] = None,
        local_files_only: bool = True,
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
        self.local_files_only = local_files_only
        self.device_map = device_map
        self.dtype = dtype
        self.attn_implementation = attn_implementation
        self.trust_remote_code = trust_remote_code

        self._model = None
        self._processor = None
        self._resolved_local_path: Optional[str] = None

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

    def _lazy_load(self) -> None:
        if self._model is not None and self._processor is not None:
            return

        local_path = self._resolve_local_path()
        logger.info("Loading Qwen3-VL model from: %s", local_path)

        from transformers import AutoProcessor
        try:
            from transformers import AutoModelForImageTextToText
        except ImportError:
            from transformers import Qwen2VLForConditionalGeneration as AutoModelForImageTextToText

        kwargs: Dict[str, Any] = {
            "device_map": self.device_map,
            "local_files_only": self.local_files_only,
            "trust_remote_code": self.trust_remote_code,
        }
        if self.attn_implementation:
            kwargs["attn_implementation"] = self.attn_implementation

        try:
            kwargs["dtype"] = self.dtype
            self._model = AutoModelForImageTextToText.from_pretrained(local_path, **kwargs)
        except TypeError:
            import torch
            kwargs.pop("dtype", None)
            if self.dtype == "auto":
                kwargs["torch_dtype"] = "auto"
            else:
                if not hasattr(torch, self.dtype):
                    raise ValueError(f"Unknown dtype={self.dtype}, use e.g. 'float16','bfloat16','auto'")
                kwargs["torch_dtype"] = getattr(torch, self.dtype)
            self._model = AutoModelForImageTextToText.from_pretrained(local_path, **kwargs)

        self._processor = AutoProcessor.from_pretrained(
            local_path,
            local_files_only=self.local_files_only,
            trust_remote_code=self.trust_remote_code,
        )
        logger.info("Qwen3-VL ready: %s", local_path)

    @property
    def resolved_local_path(self) -> str:
        return self._resolve_local_path()

    @property
    def model(self):
        self._lazy_load()
        return self._model

    @property
    def processor(self):
        self._lazy_load()
        return self._processor

    def chat(
        self,
        messages: List[Dict[str, Any]],
        gen: Optional[Qwen3VLGenConfig] = None,
    ) -> str:
        self._lazy_load()
        assert self._model is not None and self._processor is not None

        if gen is None:
            gen = Qwen3VLGenConfig()

        inputs = self._processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs.pop("token_type_ids", None)

        import torch

        try:
            with torch.inference_mode():
                device = next(iter(self._model.parameters())).device
                inputs = inputs.to(device)

                do_sample = gen.do_sample
                if do_sample is None:
                    do_sample = gen.temperature is not None and gen.temperature > 0

                generate_kwargs: Dict[str, Any] = {
                    "max_new_tokens": int(gen.max_new_tokens),
                    "do_sample": bool(do_sample),
                }
                if do_sample:
                    generate_kwargs["temperature"] = float(gen.temperature)
                    generate_kwargs["top_p"] = float(gen.top_p)

                generated_ids = self._model.generate(**inputs, **generate_kwargs)
                generated_ids_trimmed = [
                    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
                ]

                out_text = self._processor.batch_decode(
                    generated_ids_trimmed,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )
                return out_text[0] if out_text else ""
        except Exception as e:
            logger.exception("Qwen3-VL inference failed: local_path=%s", self.resolved_local_path)
            raise RuntimeError(f"Qwen3-VL inference failed (local_path={self.resolved_local_path}): {e}") from e
