# agri_rag/embedder.py
from __future__ import annotations

from typing import Any, List, Optional
import logging
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

from .image_utils import nir_to_rgb

logger = logging.getLogger(__name__)


class ClipEmbedder:
    def __init__(self, model_name: str, device: Optional[str] = None, use_fast_processor: bool = True):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        logger.info("Loading CLIP model: %s device=%s", model_name, self.device)
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

        self.processor = CLIPProcessor.from_pretrained(model_name, use_fast=use_fast_processor)

    def _ensure_tensor(self, x: Any) -> torch.Tensor:
        if torch.is_tensor(x):
            return x

        if hasattr(x, "image_embeds") and getattr(x, "image_embeds") is not None:
            return getattr(x, "image_embeds")
        if hasattr(x, "text_embeds") and getattr(x, "text_embeds") is not None:
            return getattr(x, "text_embeds")

        if hasattr(x, "pooler_output") and getattr(x, "pooler_output") is not None:
            return getattr(x, "pooler_output")

        if hasattr(x, "last_hidden_state") and getattr(x, "last_hidden_state") is not None:
            lhs = getattr(x, "last_hidden_state")
            if torch.is_tensor(lhs):
                return lhs[:, 0, :]
            return torch.as_tensor(lhs)

        if isinstance(x, (tuple, list)) and len(x) > 0:
            return self._ensure_tensor(x[0])

        raise TypeError(f"Cannot convert output of type {type(x)} to torch.Tensor")

    @torch.inference_mode()
    def _l2norm(self, x: Any) -> torch.Tensor:
        t = self._ensure_tensor(x)
        return t / (t.norm(dim=-1, keepdim=True) + 1e-12)

    @torch.inference_mode()
    def embed_image_rgb(self, rgb: Image.Image) -> List[float]:
        inputs = self.processor(images=rgb, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(self.device)

        feats = self.model.get_image_features(pixel_values=pixel_values)
        feats = self._l2norm(feats)

        return feats[0].detach().cpu().tolist()

    @torch.inference_mode()
    def embed_text(self, text: str) -> List[float]:
        inputs = self.processor(text=[text], return_tensors="pt", padding=True, truncation=True)
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)

        feats = self.model.get_text_features(input_ids=input_ids, attention_mask=attention_mask)
        feats = self._l2norm(feats)

        return feats[0].detach().cpu().tolist()

    @torch.inference_mode()
    def embed_image_rgb_nir(self, rgb: Image.Image, nir_gray: Image.Image) -> List[float]:
        e_rgb = torch.tensor(self.embed_image_rgb(rgb), device=self.device)
        nir_rgb = nir_to_rgb(nir_gray)
        e_nir = torch.tensor(self.embed_image_rgb(nir_rgb), device=self.device)

        e = (e_rgb + e_nir) / 2.0
        e = self._l2norm(e.unsqueeze(0))[0]
        return e.detach().cpu().tolist()
