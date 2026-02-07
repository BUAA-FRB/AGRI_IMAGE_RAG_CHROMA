from __future__ import annotations
import glob
import json
import os
import re
from typing import Dict, Iterable, List, Optional
from .image_utils import file_exists, mask_positive_area, safe_join
from .types import TileRecord

'''
数据扫描 + 元信息汇总 : 为每张 RAG tile 生成一条 TileRecord 记录，包括
1.图像在本地的路径
2.所属类别 train / val / test
3.NIR Mask
4.各类灾害的 label 与面积元
'''

# -------------------- 工具方法 --------------------

def _strip_id(x: str) -> str:
    """Normalize id: remove dirs and extension (e.g., a/b/c/xxx.jpg -> xxx)."""
    s = str(x).replace("\\", "/")
    base = os.path.basename(s) # 去掉目录
    return os.path.splitext(base)[0] # 去掉扩展名


def _field_id_from_tile_id(tile_id: str) -> str:
    """
    Image tile id looks like: FIELDID_x1-y1-x2-y2
    Splits json often stores only FIELDID.
    """
    s = str(tile_id)
    return s.split("_", 1)[0] if "_" in s else s # 只按第一个 _ 分割，取前缀


def _find_first_by_ext(dir_path: str, stem: str, exts: List[str]) -> Optional[str]:
    """Try <dir>/<stem><ext> for ext in exts; return first existing abs path."""
    if not dir_path or not os.path.isdir(dir_path):
        return None
    for ext in exts: 
        p = os.path.join(dir_path, stem + ext)
        if os.path.exists(p):
            return os.path.abspath(p) # 统一图像扩展名 png 和 jpg
    return None


# -------------------- 定位数据集关键文件与目录 --------------------

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

    # common: dataset_root/data2018_miniscale/field_images/rgb
    for depth in [1, 2, 3]:
        pattern = os.path.join(dataset_root, *["*"] * depth, "field_images", "rgb")
        for rgb_dir in glob.glob(pattern):
            inner = os.path.dirname(os.path.dirname(rgb_dir))  # .../<inner>/field_images
            inner = os.path.dirname(inner)  # .../<inner>
            return os.path.abspath(inner)

    raise FileNotFoundError(
        f"Cannot find 'field_images/rgb' under: {dataset_root}. "
        f"Make sure your dataset follows the Agriculture-Vision style layout."
    )


def _parse_year_from_name(path: str) -> Optional[str]:
    m = re.search(r"data(\d{4})", os.path.basename(path))
    return m.group(1) if m else None


def _load_splits(splits_json_path: str) -> Dict[str, str]:
    """
    Return mapping id -> split ('train'/'val'/'test').
    Supports:
    1) {"train":[...], "val":[...], "test":[...]}
    2) {"<id>":"train", ...}
    We normalize items by stripping extension/path.
    Note: many datasets store FIELD_ID only; we handle that later by fallback matching.
    """
    with open(splits_json_path, "r", encoding="utf-8") as f:
        obj = json.load(f)

    mapping: Dict[str, str] = {}

    # format 1
    if isinstance(obj, dict) and all(k in obj for k in ("train", "val", "test")):
        for sp in ("train", "val", "test"):
            for tid in obj.get(sp, []):
                mapping[_strip_id(str(tid))] = sp
        return mapping

    # format 2
    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(v, str) and v in ("train", "val", "test"):
                mapping[_strip_id(str(k))] = v
        if mapping:
            return mapping

    return {}


def _try_load_field_stats(inner_root: str) -> Optional[Dict[str, Dict]]:
    """
    Attempt to load per-tile stats from field_stats.json if present.
    Returns mapping tile_id(or field_id) -> stats dict.
    """
    cand = os.path.join(inner_root, "field_stats.json")
    if not os.path.exists(cand):
        return None

    try:
        with open(cand, "r", encoding="utf-8") as f:
            obj = json.load(f)
    except Exception:
        return None

    out: Dict[str, Dict] = {}

    # common: {"train": {"<tile_id>": {...}}, "val": {...}, "test": {...}}
    if isinstance(obj, dict) and any(k in obj for k in ("train", "val", "test")):
        for sp in ("train", "val", "test"):
            block = obj.get(sp)
            if isinstance(block, dict):
                for tid, stats in block.items():
                    if isinstance(stats, dict):
                        out[_strip_id(str(tid))] = stats
        return out if out else None

    # sometimes: {"<tile_id>": {...}}
    if isinstance(obj, dict) and all(isinstance(v, dict) for v in obj.values()):
        for tid, stats in obj.items():
            out[_strip_id(str(tid))] = stats
        return out if out else None

    return None


def _list_tile_ids(rgb_dir: str) -> List[str]:
    tiles: List[str] = []
    # RGB images are jpg
    for p in glob.glob(os.path.join(rgb_dir, "*.jpg")):
        tiles.append(os.path.splitext(os.path.basename(p))[0])
    tiles.sort()
    return tiles


# -------------------- 索引生成 --------------------

class AgriVisionIndex:
    """
    Build TileRecord iterator for Agriculture-Vision style layout.

    Reads:
    - splits json (train/val/test) (often FIELD_ID only)
    - field_images/rgb and (optional) field_images/nir
    - field_labels/<label> masks (optional, for metadata)
    - field_masks (optional)
    - field_stats.json (optional)
    """

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

        # common extensions for masks/labels/nir
        self.mask_exts = [".png", ".jpg", ".jpeg", ".tif", ".tiff"]

    def _get_split(self, tile_id: str) -> Optional[str]:
        """
        Try to match split by:
        1) full tile_id
        2) normalized tile_id
        3) field_id extracted from tile_id
        """
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

            # stricter filtering: if user specifies splits, require sp to exist and match
            if splits is not None:
                if sp is None or sp not in splits:
                    continue

            rgb_path = safe_join(self.rgb_dir, f"{tid}.jpg")

            # nir may not be jpg -> try multiple
            nir_path = _find_first_by_ext(self.nir_dir, tid, self.mask_exts) if os.path.isdir(self.nir_dir) else None
            nir_path = nir_path if file_exists(nir_path) else None

            # field_masks optional, try multiple extensions
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
                # try field_stats first (if present)
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
                    # fallback: scan label masks under field_labels/<label>/<tile>.(png/jpg/...)
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
