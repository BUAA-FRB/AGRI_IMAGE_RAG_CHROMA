from __future__ import annotations

import numpy as np


def compute_vari(rgb01: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    # VARI = (G - R) / (G + R - B)
    r = rgb01[..., 0]
    g = rgb01[..., 1]
    b = rgb01[..., 2]
    denom = (g + r - b)
    return (g - r) / (denom + eps)


def compute_exg(rgb01: np.ndarray) -> np.ndarray:
    # Excess Green: 2G - R - B
    r = rgb01[..., 0]
    g = rgb01[..., 1]
    b = rgb01[..., 2]
    return 2 * g - r - b


def compute_ndvi(nir01: np.ndarray, red01: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    return (nir01 - red01) / (nir01 + red01 + eps)


def vigor_proxy(rgb01: np.ndarray) -> np.ndarray:
    # A robust proxy vigor score from RGB only.
    # Combine VARI and ExG and normalize.
    vari = compute_vari(rgb01)
    exg = compute_exg(rgb01)
    # clip to reduce outliers
    vari = np.clip(vari, -1.0, 1.0)
    exg = np.clip(exg, -1.5, 1.5)
    # weighted sum
    v = 0.6 * vari + 0.4 * exg
    return v.astype(np.float32)
