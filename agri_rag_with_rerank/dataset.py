# agri_rag/dataset.py
from __future__ import annotations
import glob
import json
import os
import re
import logging
from typing import Dict, Iterable, List, Optional
from .image_utils import file_exists, mask_positive_area, safe_join
from .types import TileRecord

logger = logging.getLogger(__name__)

'''
数据扫描 + 元信息汇总 : 为每张 RAG tile 生成一条 TileRecord 记录，包括
1.图像在本地的路径
2.所属类别 train / val / test
3.NIR Mask
4.各类灾害的 label 与面积元
'''

def _strip_id(x: str) -> str:
    s = str(x).replace("\\", "/")
    base = os.path.basename(s)
    return os.path.splitext(base)[0]


def _field_id_from_tile_id(tile_id: str) -> str:
    s = str(tile_id)
    return s.split("_", 1)[0] if "_" in s else s


def _find_first_by_ext(dir_path: str, stem: str, exts: List[str]) -> Optional[str]:
    if not dir_path or not os.path.isdir(dir_path):
        return None
    for ext in exts:
        p = os.path.join(dir_path, stem + ext)
        if os.path.exists(p):
            return os.path.abspath(p)
    return None


def _find_splits_json(dataset_root: str) -> str:
    patterns = [
        os.path.join(dataset_root, "*_splits.json"),
        os.path.join(dataset_root, "*", "*_splits.json"),
        os.path.join(dataset_root, "*", "*", "*_splits.json"),
    ]
    cands: List[str] = []
    for p in patterns:
        cands.extend(glob.glob(p))
    if not cands:
        raise FileNotFoundError(f"Cannot find '*_splits.json' under: {dataset_root}")
    cands.sort(key=lambda x: len(os.path.relpath(x, dataset_root).split(os.sep)))
    return os.path.abspath(cands[0])


def _find_inner_root(dataset_root: str) -> str:
    def has_layout(root: str) -> bool:
        return os.path.isdir(os.path.join(root, "field_images", "rgb"))

    if has_layout(dataset_root):
        return os.path.abspath(dataset_root)

    for depth in [1, 2, 3]:
        pattern = os.path.join(dataset_root, *["*"] * depth, "field_images", "rgb")
        for rgb_dir in glob.glob(pattern):
            inner = os.path.dirname(os.path.dirname(rgb_dir))
            inner = os.path.dirname(inner)
            return os.path.abspath(inner)

    raise FileNotFoundError(
        f"Cannot find 'field_images/rgb' under: {dataset_root}. "
        f"Make sure your dataset follows the Agriculture-Vision style layout."
    )


def _parse_year_from_name(path: str) -> Optional[str]:
    m = re.search(r"data(\d{4})", os.path.basename(path))
    return m.group(1) if m else None


def _load_splits(splits_json_path: str) -> Dict[str, str]:
    with open(splits_json_path, "r", encoding="utf-8") as f:
        obj = json.load(f)

    mapping: Dict[str, str] = {}

    if isinstance(obj, dict) and all(k in obj for k in ("train", "val", "test")):
        for sp in ("train", "val", "test"):
            for tid in obj.get(sp, []):
                mapping[_strip_id(str(tid))] = sp
        return mapping

    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(v, str) and v in ("train", "val", "test"):
                mapping[_strip_id(str(k))] = v
        if mapping:
            return mapping

    return {}


def _try_load_field_stats(inner_root: str) -> Optional[Dict[str, Dict]]:
    cand = os.path.join(inner_root, "field_stats.json")
    if not os.path.exists(cand):
        return None

    try:
        with open(cand, "r", encoding="utf-8") as f:
            obj = json.load(f)
    except Exception:
        return None

    out: Dict[str, Dict] = {}

    if isinstance(obj, dict) and any(k in obj for k in ("train", "val", "test")):
        for sp in ("train", "val", "test"):
            block = obj.get(sp)
            if isinstance(block, dict):
                for tid, stats in block.items():
                    if isinstance(stats, dict):
                        out[_strip_id(str(tid))] = stats
        return out if out else None

    if isinstance(obj, dict) and all(isinstance(v, dict) for v in obj.values()):
        for tid, stats in obj.items():
            out[_strip_id(str(tid))] = stats
        return out if out else None

    return None


def _list_tile_ids(rgb_dir: str) -> List[str]:
    tiles: List[str] = []
    for p in glob.glob(os.path.join(rgb_dir, "*.jpg")):
        tiles.append(os.path.splitext(os.path.basename(p))[0])
    tiles.sort()
    return tiles


class AgriVisionIndex:
    def __init__(self, dataset_root: str):
        self.dataset_root = os.path.abspath(dataset_root)
        self.splits_json = _find_splits_json(self.dataset_root)
        self.inner_root = _find_inner_root(self.dataset_root)

        self.year = _parse_year_from_name(self.splits_json) or _parse_year_from_name(self.inner_root)

        self.rgb_dir = os.path.join(self.inner_root, "field_images", "rgb")
        self.nir_dir = os.path.join(self.inner_root, "field_images", "nir")
        self.field_masks_dir = os.path.join(self.inner_root, "field_masks")
        self.field_labels_dir = os.path.join(self.inner_root, "field_labels")

        self.splits = _load_splits(self.splits_json)
        self.field_stats = _try_load_field_stats(self.inner_root)

        self.label_names: List[str] = []
        if os.path.isdir(self.field_labels_dir):
            for d in os.listdir(self.field_labels_dir):
                if os.path.isdir(os.path.join(self.field_labels_dir, d)):
                    self.label_names.append(d)
        self.label_names.sort()

        self.tile_ids = _list_tile_ids(self.rgb_dir)
        self.mask_exts = [".png", ".jpg", ".jpeg", ".tif", ".tiff"]

        logger.debug("Dataset loaded: root=%s inner=%s tiles=%d splits=%s",
                     self.dataset_root, self.inner_root, len(self.tile_ids), self.splits_json)

    def _get_split(self, tile_id: str) -> Optional[str]:
        tid = _strip_id(tile_id)
        sp = self.splits.get(tid)
        if sp is not None:
            return sp
        fid = _field_id_from_tile_id(tid)
        return self.splits.get(fid)

    def iter_tiles(
        self,
        splits: Optional[List[str]] = None,
        max_items: Optional[int] = None,
        fast_metadata: bool = False,
    ) -> Iterable[TileRecord]:
        count = 0
        for tid in self.tile_ids:
            sp = self._get_split(tid)

            if splits is not None:
                if sp is None or sp not in splits:
                    continue

            rgb_path = safe_join(self.rgb_dir, f"{tid}.jpg")

            nir_path = _find_first_by_ext(self.nir_dir, tid, self.mask_exts) if os.path.isdir(self.nir_dir) else None
            nir_path = nir_path if file_exists(nir_path) else None

            field_mask_path = _find_first_by_ext(self.field_masks_dir, tid, self.mask_exts) if os.path.isdir(self.field_masks_dir) else None
            field_mask_path = field_mask_path if file_exists(field_mask_path) else None

            labels_present: List[str] = []
            label_areas: Dict[str, int] = {}

            extra: Dict = {
                "dataset_root": self.dataset_root,
                "inner_root": self.inner_root,
                "year": self.year,
                "splits_json": self.splits_json,
            }

            if not fast_metadata:
                stats = None
                if isinstance(self.field_stats, dict):
                    stats = self.field_stats.get(_strip_id(tid)) or self.field_stats.get(_field_id_from_tile_id(_strip_id(tid)))

                if isinstance(stats, dict):
                    if "label_areas" in stats and isinstance(stats["label_areas"], dict):
                        for k, v in stats["label_areas"].items():
                            try:
                                area_sum = int(sum(v)) if isinstance(v, list) else int(v)
                            except Exception:
                                continue
                            if area_sum > 0:
                                labels_present.append(str(k))
                                label_areas[str(k)] = area_sum
                    elif "label_counts" in stats and isinstance(stats["label_counts"], dict):
                        for k, v in stats["label_counts"].items():
                            try:
                                cnt = int(v)
                            except Exception:
                                continue
                            if cnt > 0:
                                labels_present.append(str(k))
                                label_areas[str(k)] = 0
                    extra["field_stats"] = stats
                else:
                    for lb in self.label_names:
                        lb_dir = os.path.join(self.field_labels_dir, lb)
                        mp = _find_first_by_ext(lb_dir, tid, self.mask_exts)
                        if not mp:
                            continue
                        area = mask_positive_area(mp, threshold=0)
                        if area > 0:
                            labels_present.append(lb)
                            label_areas[lb] = int(area)

            rec = TileRecord(
                tile_id=tid,
                split=sp,
                rgb_path=rgb_path,
                nir_path=nir_path,
                field_mask_path=field_mask_path,
                labels_present=labels_present,
                label_areas=label_areas,
                extra=extra,
            )

            yield rec
            count += 1
            if max_items is not None and count >= max_items:
                break
