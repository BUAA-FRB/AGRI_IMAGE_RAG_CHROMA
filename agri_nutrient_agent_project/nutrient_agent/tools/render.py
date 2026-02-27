from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from .image_utils import normalize01


def _apply_colormap_to_01(img01: np.ndarray, cmap_name: str = "viridis") -> np.ndarray:
    """
    img01: float32 0..1
    return: uint8 RGB image (H,W,3) without any borders/text
    """
    import matplotlib.cm as cm

    x = np.clip(np.asarray(img01, dtype=np.float32), 0.0, 1.0)
    cmap = cm.get_cmap(cmap_name)
    rgba = cmap(x)  # float 0..1, (H,W,4)
    rgb = (rgba[..., :3] * 255.0).astype(np.uint8)
    return rgb


def save_index_preview(index01: np.ndarray, out_png: Path, title: str = "Vigor Proxy") -> None:
    """
    IMPORTANT CHANGE:
    - No matplotlib figure (no padding, no title, no text)
    - Save as a pure colormapped image -> eliminates whitespace & labels
    """
    out_png.parent.mkdir(parents=True, exist_ok=True)
    img01 = normalize01(index01)
    rgb = _apply_colormap_to_01(img01, cmap_name="viridis")
    Image.fromarray(rgb, mode="RGB").save(out_png)


def save_severity_heatmap_rgba(class_map: np.ndarray, out_png: Path, alpha: float = 0.65) -> None:
    """
    class_map: 0=healthy,1=mild,2=moderate,3=severe
    Output: RGBA PNG with transparent background for class 0.
    """
    out_png.parent.mkdir(parents=True, exist_ok=True)
    H, W = class_map.shape
    rgba = np.zeros((H, W, 4), dtype=np.uint8)

    # mild: yellow, moderate: orange, severe: red
    rgba[class_map == 1] = (255, 233, 140, int(255 * (alpha * 0.55)))
    rgba[class_map == 2] = (255, 170, 80,  int(255 * (alpha * 0.75)))
    rgba[class_map == 3] = (255, 80,  80,  int(255 * (alpha * 1.00)))

    Image.fromarray(rgba, mode="RGBA").save(out_png)


# -------------------------
# richer charts (keep as-is)
# -------------------------
def save_low_vigor_histogram(
    low_vigor01: np.ndarray,
    thresholds: Dict[str, float],
    out_png: Path,
    title: str = "Low-vigor Histogram with Thresholds",
) -> None:
    out_png.parent.mkdir(parents=True, exist_ok=True)
    data = np.asarray(low_vigor01, dtype=np.float32).reshape(-1)
    data = data[np.isfinite(data)]

    fig = plt.figure(figsize=(7.2, 4.2), dpi=150)
    ax = fig.add_subplot(111)
    ax.hist(data, bins=60, alpha=0.9)
    ax.set_title(title)
    ax.set_xlabel("low_vigor (0..1)")
    ax.set_ylabel("count")

    for k, v in thresholds.items():
        ax.axvline(float(v), linestyle="--", linewidth=2, label=f"{k}={float(v):.3f}")

    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)


def save_severity_distribution_bar(
    class_map: np.ndarray,
    out_png: Path,
    title: str = "Severity Distribution (pixel counts)",
) -> Dict[str, int]:
    out_png.parent.mkdir(parents=True, exist_ok=True)
    cm = np.asarray(class_map, dtype=np.uint8)
    counts = {
        "healthy": int(np.sum(cm == 0)),
        "mild": int(np.sum(cm == 1)),
        "moderate": int(np.sum(cm == 2)),
        "severe": int(np.sum(cm == 3)),
    }

    labels = list(counts.keys())
    values = [counts[k] for k in labels]

    fig = plt.figure(figsize=(7.2, 4.2), dpi=150)
    ax = fig.add_subplot(111)
    ax.bar(labels, values)
    ax.set_title(title)
    ax.set_ylabel("pixel count")
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)
    return counts


def save_patch_area_bar(
    comps_xyxy_area: List[Tuple[int, int, int, int, int]],
    out_png: Path,
    top_k: int = 10,
    title: str = "Top Patch Areas (px)",
) -> List[int]:
    out_png.parent.mkdir(parents=True, exist_ok=True)
    areas = [int(c[4]) for c in comps_xyxy_area[:top_k]]
    if not areas:
        areas = [0]

    fig = plt.figure(figsize=(7.2, 4.2), dpi=150)
    ax = fig.add_subplot(111)
    ax.bar([f"P{i}" for i in range(len(areas))], areas)
    ax.set_title(title)
    ax.set_ylabel("area_px")
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)
    return areas