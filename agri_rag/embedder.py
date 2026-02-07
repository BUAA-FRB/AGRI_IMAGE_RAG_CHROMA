from __future__ import annotations

from typing import Any, List, Optional

import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

from .image_utils import nir_to_rgb


class ClipEmbedder:
    """
    CLIP-based embedder for image/text embeddings used in image RAG.

    Fixes:
    - Transformers may return a BaseModelOutput/CLIPOutput object instead of a Tensor
      depending on version / calling path.
    - We therefore ALWAYS convert outputs into a Tensor before L2 normalization.
    """

    def __init__(self, model_name: str, device: Optional[str] = None, use_fast_processor: bool = True):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Load model + processor (local dir or HF repo id both OK)
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

        # use_fast_processor: keep default behavior unless you want strict reproducibility
        self.processor = CLIPProcessor.from_pretrained(model_name, use_fast=use_fast_processor)

    def _ensure_tensor(self, x: Any) -> torch.Tensor:
        """
        Convert various HF output objects to a torch.Tensor.

        Possible return types from Transformers:
        - torch.Tensor
        - CLIPOutput with .image_embeds / .text_embeds
        - BaseModelOutputWithPooling with .pooler_output / .last_hidden_state
        - tuple/list of the above
        """
        if torch.is_tensor(x):
            return x

        # CLIPOutput path
        if hasattr(x, "image_embeds") and getattr(x, "image_embeds") is not None:
            return getattr(x, "image_embeds")
        if hasattr(x, "text_embeds") and getattr(x, "text_embeds") is not None:
            return getattr(x, "text_embeds")

        # BaseModelOutputWithPooling path
        if hasattr(x, "pooler_output") and getattr(x, "pooler_output") is not None:
            return getattr(x, "pooler_output")

        # BaseModelOutput path: use CLS token as fallback
        if hasattr(x, "last_hidden_state") and getattr(x, "last_hidden_state") is not None:
            lhs = getattr(x, "last_hidden_state")
            if torch.is_tensor(lhs):
                # [B, T, D] -> take CLS token at index 0
                return lhs[:, 0, :]
            return torch.as_tensor(lhs)

        # Tuple/list fallback
        if isinstance(x, (tuple, list)) and len(x) > 0:
            return self._ensure_tensor(x[0])

        raise TypeError(f"Cannot convert output of type {type(x)} to torch.Tensor")

    @torch.inference_mode()
    def _l2norm(self, x: Any) -> torch.Tensor:
        t = self._ensure_tensor(x)
        return t / (t.norm(dim=-1, keepdim=True) + 1e-12)

    @torch.inference_mode()
    def embed_image_rgb(self, rgb: Image.Image) -> List[float]:
        """
        Encode an RGB PIL image into a normalized embedding vector.
        """
        inputs = self.processor(images=rgb, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(self.device)

        # Preferred stable API
        feats = self.model.get_image_features(pixel_values=pixel_values)
        feats = self._l2norm(feats)

        return feats[0].detach().cpu().tolist()

    @torch.inference_mode()
    def embed_text(self, text: str) -> List[float]:
        """
        Encode a text query into a normalized embedding vector.
        """
        inputs = self.processor(text=[text], return_tensors="pt", padding=True, truncation=True)
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)

        feats = self.model.get_text_features(input_ids=input_ids, attention_mask=attention_mask)
        feats = self._l2norm(feats)

        return feats[0].detach().cpu().tolist()

    @torch.inference_mode()
    def embed_image_rgb_nir(self, rgb: Image.Image, nir_gray: Image.Image) -> List[float]:
        """
        Fuse RGB and NIR embeddings: avg + renorm.
        NIR gray is converted to a 3-channel image via nir_to_rgb.
        """
        e_rgb = torch.tensor(self.embed_image_rgb(rgb), device=self.device)
        nir_rgb = nir_to_rgb(nir_gray)
        e_nir = torch.tensor(self.embed_image_rgb(nir_rgb), device=self.device)

        e = (e_rgb + e_nir) / 2.0
        e = self._l2norm(e.unsqueeze(0))[0]
        return e.detach().cpu().tolist()
