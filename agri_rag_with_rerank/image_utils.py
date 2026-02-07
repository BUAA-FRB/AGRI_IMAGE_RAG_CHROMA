# agri_rag/image_utils.py
from __future__ import annotations
import os
from typing import Optional, Tuple
import numpy as np
from PIL import Image


def open_rgb(path: str) -> Image.Image:
    img = Image.open(path)
    if img.mode != "RGB":
        img = img.convert("RGB")
    return img


def open_gray(path: str) -> Image.Image:
    img = Image.open(path)
    if img.mode != "L":
        img = img.convert("L")
    return img


def nir_to_rgb(nir_gray: Image.Image) -> Image.Image:
    if nir_gray.mode != "L":
        nir_gray = nir_gray.convert("L")
    return Image.merge("RGB", (nir_gray, nir_gray, nir_gray))


def mask_has_positive(path: str, threshold: int = 0) -> bool:
    m = open_gray(path)
    arr = np.array(m)
    return bool((arr > threshold).any())


def mask_positive_area(path: str, threshold: int = 0) -> int:
    m = open_gray(path)
    arr = np.array(m)
    return int((arr > threshold).sum())


def safe_join(*parts: str) -> str:
    return os.path.abspath(os.path.join(*parts))


def file_exists(path: Optional[str]) -> bool:
    return path is not None and os.path.exists(path)
