from typing import List, Optional
import os


class TextEmbedder:
    """
    BGE 文本向量：transformers + CLS pooling + L2 normalize
    - 优先使用 safetensors（避免 torch.load 的版本限制问题）
    - CUDA 时默认 fp16（更省显存）
    """

    def __init__(
        self,
        model_name_or_path: str,
        local_files_only: bool = True,
        device: Optional[str] = None,
        max_length: int = 512,
        batch_size: int = 32,
    ):
        self.model = model_name_or_path
        self.local_files_only = local_files_only
        self.max_length = int(max_length)
        self.batch_size = int(batch_size)

        if os.path.exists(self.model) and self.local_files_only:
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
            os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

        self._tok = None
        self._model = None
        self._device = device  # may be None

    def _lazy_load(self):
        if self._tok is not None and self._model is not None:
            return

        from transformers import AutoTokenizer, AutoModel  # type: ignore
        import torch

        self._tok = AutoTokenizer.from_pretrained(
            self.model,
            local_files_only=self.local_files_only,
            trust_remote_code=True,
        )

        # 关键：优先 safetensors，避免 torch.load 触发版本限制
        model_kwargs = dict(
            local_files_only=self.local_files_only,
            trust_remote_code=True,
        )
        # CUDA 用 fp16 降显存
        if self._device is None:
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
        if self._device == "cuda":
            model_kwargs["torch_dtype"] = torch.float16

        # transformers 不同版本对 use_safetensors 支持不同，做兼容
        try:
            model_kwargs["use_safetensors"] = True
            self._model = AutoModel.from_pretrained(self.model, **model_kwargs)
        except TypeError:
            model_kwargs.pop("use_safetensors", None)
            self._model = AutoModel.from_pretrained(self.model, **model_kwargs)

        self._model.eval()
        self._model.to(torch.device(self._device))

    @staticmethod
    def _l2_normalize_batch(x):
        import torch
        return torch.nn.functional.normalize(x, p=2, dim=1)

    def _embed_batch(self, batch_texts: List[str]) -> List[List[float]]:
        self._lazy_load()
        import torch

        inputs = self._tok(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )
        dev = next(self._model.parameters()).device
        inputs = {k: v.to(dev) for k, v in inputs.items()}

        with torch.inference_mode():
            out = self._model(**inputs)

        emb = out.last_hidden_state[:, 0, :]  # CLS
        emb = self._l2_normalize_batch(emb)

        embs = emb.detach().to(torch.float32).cpu().numpy()
        return [e.tolist() for e in embs]

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        texts = [t if isinstance(t, str) else str(t) for t in texts]
        if not texts:
            return []

        out: List[List[float]] = []
        bs = max(1, int(self.batch_size))
        for i in range(0, len(texts), bs):
            out.extend(self._embed_batch(texts[i:i + bs]))
        return out
