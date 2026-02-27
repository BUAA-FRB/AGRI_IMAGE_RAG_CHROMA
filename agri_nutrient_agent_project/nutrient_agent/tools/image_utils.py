from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, List

import numpy as np
from PIL import Image, ImageDraw


@dataclass
class LoadedImage:
    path: Path
    pil: Image.Image
    np_rgb: np.ndarray  # float32 in [0,1], shape (H,W,3)


def load_rgb(path: Path) -> LoadedImage:
    img = Image.open(path).convert("RGB")
    arr = np.asarray(img).astype(np.float32) / 255.0
    return LoadedImage(path=path, pil=img, np_rgb=arr)


def load_gray(path: Path) -> np.ndarray:
    img = Image.open(path).convert("L")
    arr = np.asarray(img).astype(np.float32) / 255.0
    return arr


def parse_bbox_str(bbox: str) -> Optional[Tuple[int, int, int, int]]:
    if not bbox:
        return None
    s = bbox.strip()
    if not (s.startswith("(") and s.endswith(")")):
        return None
    parts = [p.strip() for p in s[1:-1].split(",")]
    if len(parts) != 4:
        return None
    try:
        x0, y0, x1, y1 = [int(float(p)) for p in parts]
        return x0, y0, x1, y1
    except Exception:
        return None


def crop_pil(img: Image.Image, bbox: Tuple[int, int, int, int]) -> Image.Image:
    x0, y0, x1, y1 = bbox
    x0 = max(0, min(x0, img.width))
    x1 = max(0, min(x1, img.width))
    y0 = max(0, min(y0, img.height))
    y1 = max(0, min(y1, img.height))
    if x1 <= x0 or y1 <= y0:
        return img.copy()
    return img.crop((x0, y0, x1, y1))


def draw_bbox(img: Image.Image, bbox: Tuple[int, int, int, int], label: str = "") -> Image.Image:
    out = img.copy()
    d = ImageDraw.Draw(out)
    x0, y0, x1, y1 = bbox
    d.rectangle([x0, y0, x1, y1], outline=(255, 0, 0), width=3)
    if label:
        d.text((x0 + 3, y0 + 3), label, fill=(255, 0, 0))
    return out


def save_image(img: Image.Image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path)


def normalize01(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    mn = float(np.nanmin(x))
    mx = float(np.nanmax(x))
    if mx - mn < 1e-8:
        return np.zeros_like(x, dtype=np.float32)
    return (x - mn) / (mx - mn)


def rgba_to_rgb_white(im: Image.Image) -> Image.Image:
    """Convert RGBA to RGB by compositing on white background."""
    if im.mode != "RGBA":
        return im.convert("RGB")
    bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
    comp = Image.alpha_composite(bg, im)
    return comp.convert("RGB")


# -------------------------
# Uniform montage utils
# -------------------------
def fit_cover_and_center_crop(im: Image.Image, target_w: int, target_h: int) -> Image.Image:
    """Resize (keep aspect) to cover target and then center-crop to exact size."""
    if im.mode not in ("RGB", "RGBA"):
        im = im.convert("RGB")
    w, h = im.size
    if w <= 0 or h <= 0:
        return Image.new("RGB", (target_w, target_h), (240, 240, 240))

    scale = max(target_w / w, target_h / h)
    new_w = max(target_w, int(round(w * scale)))
    new_h = max(target_h, int(round(h * scale)))
    resized = im.resize((new_w, new_h), resample=Image.BICUBIC)

    left = max(0, (new_w - target_w) // 2)
    top = max(0, (new_h - target_h) // 2)
    return resized.crop((left, top, left + target_w, top + target_h))


def make_uniform_grid_montage(
    images: List[Image.Image],
    cols: int = 3,
    rows: int = 2,
    cell_size: Tuple[int, int] = (320, 320),
    pad: int = 14,
    bg: Tuple[int, int, int] = (245, 245, 245),
    force_rgb: bool = True,
    border_px: int = 3,  # NEW
    border_color: Tuple[int, int, int] = (0, 0, 0),  # NEW
) -> Image.Image:
    """
    Create a rows×cols montage with identical cell size (no distortion),
    and draw a black border for each cell (helps distinguish light images).
    """
    tw, th = cell_size
    W = cols * tw + (cols + 1) * pad
    H = rows * th + (rows + 1) * pad
    canvas = Image.new("RGB", (W, H), bg)
    draw = ImageDraw.Draw(canvas)

    max_n = rows * cols
    for i in range(min(len(images), max_n)):
        r = i // cols
        c = i % cols
        x = pad + c * (tw + pad)
        y = pad + r * (th + pad)

        cell = fit_cover_and_center_crop(images[i], tw, th)
        if force_rgb:
            cell = rgba_to_rgb_white(cell)
        canvas.paste(cell, (x, y))

        # NEW: border
        if border_px > 0:
            for k in range(border_px):
                draw.rectangle(
                    [x + k, y + k, x + tw - 1 - k, y + th - 1 - k],
                    outline=border_color
                )

    return canvas