from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image, ImageFilter

from .image_utils import normalize01


@dataclass
class SeverityResult:
    vigor01: np.ndarray          # (H,W) float32 0..1 (higher=healthier)
    low_vigor01: np.ndarray      # (H,W) float32 0..1 (higher=worse)
    class_map: np.ndarray        # (H,W) uint8: 0=healthy,1=mild,2=moderate,3=severe
    thresholds: Dict[str, float]
    affected_mask: np.ndarray    # (H,W) bool (moderate+severe)


def compute_severity_from_vigor(vigor: np.ndarray, mild_p: float, mod_p: float, sev_p: float) -> SeverityResult:
    # vigor: higher=healthier. We invert to low_vigor (worse)
    v01 = normalize01(vigor)
    low = 1.0 - v01

    # smooth a bit for stability (PIL box blur)
    low_img = Image.fromarray((low * 255).astype(np.uint8))
    low_img = low_img.filter(ImageFilter.BoxBlur(2))
    low_s = np.asarray(low_img).astype(np.float32) / 255.0

    t_mild = float(np.quantile(low_s, mild_p))
    t_mod = float(np.quantile(low_s, mod_p))
    t_sev = float(np.quantile(low_s, sev_p))

    cm = np.zeros_like(low_s, dtype=np.uint8)
    cm[low_s >= t_mild] = 1
    cm[low_s >= t_mod] = 2
    cm[low_s >= t_sev] = 3

    affected = (cm >= 2)

    return SeverityResult(
        vigor01=v01.astype(np.float32),
        low_vigor01=low_s.astype(np.float32),
        class_map=cm,
        thresholds={"mild": t_mild, "moderate": t_mod, "severe": t_sev},
        affected_mask=affected
    )


def connected_components(mask: np.ndarray, min_area: int = 150) -> List[Tuple[int,int,int,int,int]]:
    # Returns list of components as (x0,y0,x1,y1,area) using BFS over pixels.
    # For 512x512 it's ok.
    H, W = mask.shape
    visited = np.zeros_like(mask, dtype=np.uint8)
    comps: List[Tuple[int,int,int,int,int]] = []
    for y in range(H):
        row = mask[y]
        for x in range(W):
            if row[x] and not visited[y, x]:
                # BFS
                q = [(x, y)]
                visited[y, x] = 1
                x0=x1=x
                y0=y1=y
                area=0
                while q:
                    cx, cy = q.pop()
                    area += 1
                    if cx < x0: x0 = cx
                    if cx > x1: x1 = cx
                    if cy < y0: y0 = cy
                    if cy > y1: y1 = cy
                    # 4-neighborhood
                    if cx > 0 and mask[cy, cx-1] and not visited[cy, cx-1]:
                        visited[cy, cx-1]=1; q.append((cx-1, cy))
                    if cx+1 < W and mask[cy, cx+1] and not visited[cy, cx+1]:
                        visited[cy, cx+1]=1; q.append((cx+1, cy))
                    if cy > 0 and mask[cy-1, cx] and not visited[cy-1, cx]:
                        visited[cy-1, cx]=1; q.append((cx, cy-1))
                    if cy+1 < H and mask[cy+1, cx] and not visited[cy+1, cx]:
                        visited[cy+1, cx]=1; q.append((cx, cy+1))
                if area >= min_area:
                    # inclusive bbox -> make x1/y1 exclusive
                    comps.append((x0, y0, x1+1, y1+1, area))
    # sort by area desc
    comps.sort(key=lambda t: t[4], reverse=True)
    return comps
