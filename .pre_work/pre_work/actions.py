from __future__ import annotations
from typing import Dict, List

def suggest_actions_for_day(levels: Dict[str, str], triggers: Dict[str, str]) -> List[str]:
    out: List[str] = []
    if levels.get("waterlogging") in ("L2","L3"):
        out.append("P0 排涝/疏渠：优先检查入水口/弯道/低洼段；必要时临时抽排。" + (f"（触发：{triggers.get('waterlogging','')}）" if triggers.get("waterlogging") else ""))
    if levels.get("drought") in ("L2","L3"):
        out.append("P0 灌溉窗口：优先保障关键生育期；采用分区轮灌/滴灌，避免一次性大水造成倒伏。" + (f"（触发：{triggers.get('drought','')}）" if triggers.get("drought") else ""))
    if levels.get("heat") in ("L2","L3"):
        out.append("P1 高温应对：避开正午作业；结合灌溉降温/叶面保护，重点关注开花灌浆期。" + (f"（触发：{triggers.get('heat','')}）" if triggers.get("heat") else ""))
    if levels.get("frost") in ("L2","L3"):
        out.append("P0 霜冻预案：夜间巡查，必要时覆盖/防霜灌溉/熏烟（遵守当地规定）。" + (f"（触发：{triggers.get('frost','')}）" if triggers.get("frost") else ""))
    if levels.get("disease_pressure") in ("L2","L3"):
        out.append("P1 病害窗口：雨后24–48h加强田间巡检；结合历史病害与药剂方案做预防性管理。" + (f"（触发：{triggers.get('disease_pressure','')}）" if triggers.get("disease_pressure") else ""))
    if levels.get("operation_window") in ("L2","L3"):
        out.append("P2 例行作业：适合开展巡检、补沟、轻量机械作业与台账更新。")
    return out
