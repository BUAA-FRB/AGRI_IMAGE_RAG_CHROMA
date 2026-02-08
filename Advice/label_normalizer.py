import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple


def _norm(s: str) -> str:
    # 统一成小写；去空格；把连字符/空格等归一
    x = (s or "").strip().lower()
    x = x.replace(" ", "_").replace("-", "_")
    x = x.replace("__", "_")
    return x


@dataclass
class LabelSpec:
    canonical: str
    aliases: List[str]
    keywords: List[str]


def default_label_specs() -> Dict[str, LabelSpec]:
    """
    以你给的 canonical label 为准，同时加常见别名/拼写纠错/中英文关键词。
    """
    specs: Dict[str, LabelSpec] = {}

    def add(canon: str, aliases: List[str], keywords: List[str]):
        canon_n = _norm(canon)
        specs[canon_n] = LabelSpec(
            canonical=canon_n,
            aliases=[_norm(a) for a in aliases if a],
            keywords=[k.strip() for k in keywords if str(k).strip()],
        )

    add(
        "double_plant",
        aliases=["doubleplant", "double-plant", "double planting", "double_planting"],
        keywords=["重播", "重苗", "双株", "重复播种", "播种过密", "double plant", "double planting"],
    )
    add(
        "drydown",
        aliases=["dry_down", "dry-down", "drought", "drydown"],
        keywords=["干旱", "缺水", "失墒", "土壤墒情", "干热风", "drought", "water stress"],
    )
    add(
        "endrow",
        aliases=["end_row", "end-row", "headland", "field_edge", "border"],
        keywords=["地头", "田头", "地块边缘", "边行", "行末", "headland", "end row"],
    )
    # 你最终标准写的是 nurient_deficiency（拼写少了 t），我们把它当 canonical
    add(
        "nurient_deficiency",
        aliases=[
            "nutrient_deficiency", "nutrient-deficiency", "nutrient deficiency",
            "nutrition_deficiency", "nutrient_lack"
        ],
        keywords=["缺素", "养分不足", "营养缺乏", "黄化", "叶脉间失绿", "nutrient deficiency"],
    )
    add(
        "planter_skip",
        aliases=["planter-skip", "skip", "miss_plant", "missing_plants", "stand_gap"],
        keywords=["漏播", "缺苗", "断垄", "出苗不齐", "播种缺失", "stand gap", "skip"],
    )
    add(
        "storm_damage",
        aliases=["storm-damage", "wind_damage", "hail_damage", "typhoon_damage"],
        keywords=["风害", "倒伏", "冰雹", "暴雨", "台风", "强对流", "storm damage", "wind damage"],
    )
    add(
        "water",
        aliases=["standing_water", "ponding", "flood", "surface_water"],
        keywords=["积水", "水体", "水面", "洪涝", "田间积水", "standing water", "ponding"],
    )
    # 你最终标准写的是 water_way，我们把它当 canonical
    add(
        "water_way",
        aliases=["waterway", "water-way", "ditch", "canal", "channel", "irrigation_ditch"],
        keywords=["沟渠", "水道", "渠系", "灌溉渠", "排水沟", "渠道", "ditch", "canal", "waterway"],
    )
    # 你前面提过 weed_cluster，这里也加上，后面不用可忽略
    add(
        "weed_cluster",
        aliases=["weedcluster", "weed_patch", "weed-cluster", "weeds"],
        keywords=["杂草", "杂草丛", "杂草斑块", "杂草聚集", "weed patch", "weed cluster"],
    )

    return specs


def load_label_specs(label_map_path: Optional[str]) -> Dict[str, LabelSpec]:
    """
    支持外部 json 覆盖/扩展。
    文件格式推荐：
    {
      "water_way": {"aliases": ["waterway"], "keywords": ["沟渠","水道"]},
      "nurient_deficiency": {"aliases": ["nutrient_deficiency"], "keywords": ["缺素","黄化"]}
    }
    """
    base = default_label_specs()

    if not label_map_path:
        return base

    p = os.path.abspath(label_map_path)
    if not os.path.exists(p):
        return base

    try:
        with open(p, "r", encoding="utf-8") as f:
            obj = json.load(f)
    except Exception:
        return base

    if not isinstance(obj, dict):
        return base

    for canon, spec in obj.items():
        canon_n = _norm(str(canon))
        if canon_n not in base:
            base[canon_n] = LabelSpec(canonical=canon_n, aliases=[], keywords=[])

        if isinstance(spec, dict):
            aliases = spec.get("aliases") or []
            keywords = spec.get("keywords") or []
            if isinstance(aliases, str):
                aliases = [aliases]
            if isinstance(keywords, str):
                keywords = [keywords]
            base[canon_n].aliases.extend([_norm(str(a)) for a in aliases if str(a).strip()])
            base[canon_n].keywords.extend([str(k).strip() for k in keywords if str(k).strip()])

    # 去重
    for k, v in base.items():
        v.aliases = list(dict.fromkeys([_norm(x) for x in v.aliases if x]))
        v.keywords = list(dict.fromkeys([x for x in v.keywords if x]))

    return base


class LabelNormalizer:
    def __init__(self, label_map_path: Optional[str] = None):
        self.specs = load_label_specs(label_map_path)

        # alias -> canonical
        self.alias2canon: Dict[str, str] = {}
        for canon, spec in self.specs.items():
            self.alias2canon[_norm(canon)] = canon
            for a in spec.aliases:
                self.alias2canon[_norm(a)] = canon

    def normalize_one(self, label: str) -> str:
        x = _norm(label)
        return self.alias2canon.get(x, x)

    def normalize_many(self, labels: List[str]) -> List[str]:
        out: List[str] = []
        for l in labels or []:
            c = self.normalize_one(str(l))
            if c and c not in out:
                out.append(c)
        return out

    def expand_for_retrieval(self, canonical_labels: List[str]) -> Tuple[List[str], List[str]]:
        """
        返回：(expanded_tags, expanded_keywords)
        expanded_tags 用于 tag_boost（包含 canonical + aliases）
        expanded_keywords 用于 query 文本补充（中文/英文关键词）
        """
        tags: List[str] = []
        kws: List[str] = []

        for c in self.normalize_many(canonical_labels):
            if c not in tags:
                tags.append(c)
            spec = self.specs.get(c)
            if not spec:
                continue

            for a in spec.aliases:
                if a and a not in tags:
                    tags.append(a)

            for k in spec.keywords:
                if k and k not in kws:
                    kws.append(k)

        return tags, kws
