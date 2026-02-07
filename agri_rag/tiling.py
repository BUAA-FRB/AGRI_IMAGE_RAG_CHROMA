from __future__ import annotations
from typing import List, Tuple
from PIL import Image
from .types import BBox


def tile_image(img: Image.Image, tile: int = 224, stride: int = 160) -> List[Tuple[Image.Image, BBox]]:
    """
    Sliding window tiling.
    Returns list of (patch_image, bbox=(x1,y1,x2,y2)) in original image coords.
    """
    w, h = img.size
    patches: List[Tuple[Image.Image, BBox]] = []

    if w < tile or h < tile:
        patches.append((img.resize((tile, tile)), (0, 0, w, h)))
        return patches

    for y in range(0, h - tile + 1, stride):
        for x in range(0, w - tile + 1, stride):
            bbox = (x, y, x + tile, y + tile)
            patches.append((img.crop(bbox), bbox))
    return patches
