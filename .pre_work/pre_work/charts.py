from __future__ import annotations
from pathlib import Path
from typing import Dict, List
import numpy as np
import matplotlib.pyplot as plt

def _cn_font() -> None:
    try:
        plt.rcParams["font.sans-serif"] = ["Microsoft YaHei","SimHei","Arial Unicode MS","DejaVu Sans"]
        plt.rcParams["axes.unicode_minus"] = False
    except Exception:
        pass

def _save(fig, out_png: Path) -> None:
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_png, dpi=220)
    plt.close(fig)

def plot_risk_curves(dates: List[str], risk: Dict[str, List[float]], out_png: Path) -> None:
    _cn_font()
    x = np.arange(len(dates))
    fig = plt.figure(figsize=(11, 6.2))

    ax1 = fig.add_subplot(2, 1, 1)
    keys = ["waterlogging","drought","heat","frost","disease_pressure","wind_lodging"]
    for k in keys:
        if k in risk and risk[k]:
            ax1.plot(x, risk[k], label=k)
    for thr in (35,60,80):
        ax1.axhline(thr, linestyle="--", linewidth=1.0, alpha=0.6)
    ax1.set_title("多风险曲线（0-100），虚线为 L1/L2/L3 阈值")
    ax1.set_ylabel("风险分")
    ax1.set_xlim(0, max(0, len(dates)-1))
    ax1.grid(True, alpha=0.2)
    ax1.legend(ncol=3, fontsize=9, frameon=False)

    ax2 = fig.add_subplot(2, 1, 2, sharex=ax1)
    if "operation_window" in risk and risk["operation_window"]:
        ax2.plot(x, risk["operation_window"], label="operation_window")
        ax2.axhline(60, linestyle="--", linewidth=1.0, alpha=0.6)
        ax2.set_title("作业窗口指数（越高越适合巡检/灌排等）")
        ax2.set_ylabel("指数")
        ax2.grid(True, alpha=0.2)
        ax2.legend(frameon=False)

    step = max(1, len(dates)//10)
    ax2.set_xticks(x[::step])
    ax2.set_xticklabels([dates[i] for i in range(0, len(dates), step)], rotation=30, ha="right")
    ax2.set_xlabel("日期")
    _save(fig, out_png)

def plot_water_balance(dates: List[str], ind: Dict[str, List[float]], out_png: Path) -> None:
    _cn_font()
    x = np.arange(len(dates))
    p = np.array(ind.get("precip_mm", [0.0]*len(dates)), dtype=float)
    et0 = np.array(ind.get("et0_mm", [0.0]*len(dates)), dtype=float)
    wb7 = np.array(ind.get("water_balance_7d_mm", [0.0]*len(dates)), dtype=float)

    fig = plt.figure(figsize=(11, 5.6))
    ax = fig.add_subplot(1, 1, 1)
    ax.bar(x, p, alpha=0.6, label="降水(mm)")
    ax.plot(x, et0, label="ET0(mm)")
    ax.plot(x, wb7, label="7日水分收支(降水-ET0, mm)", linewidth=2.0)
    ax.axhline(0, linewidth=1.0, alpha=0.7)
    ax.set_title("降水 / ET0 / 水分收支（用于提前识别涝与旱的趋势）")
    ax.set_ylabel("mm")
    ax.grid(True, alpha=0.2)
    ax.legend(ncol=3, frameon=False, fontsize=9)

    step = max(1, len(dates)//10)
    ax.set_xticks(x[::step])
    ax.set_xticklabels([dates[i] for i in range(0, len(dates), step)], rotation=30, ha="right")
    ax.set_xlabel("日期")
    _save(fig, out_png)

def plot_soil_moisture(dates: List[str], ind: Dict[str, List[float]], out_png: Path) -> None:
    _cn_font()
    x = np.arange(len(dates))
    sm0 = np.array(ind.get("soil_moisture_0_7", [0.0]*len(dates)), dtype=float)
    sm28 = np.array(ind.get("soil_moisture_7_28", [0.0]*len(dates)), dtype=float)
    sm100 = np.array(ind.get("soil_moisture_0_100", [0.0]*len(dates)), dtype=float)
    z0 = np.array(ind.get("soil_moisture_0_7_z", [0.0]*len(dates)), dtype=float)
    z100 = np.array(ind.get("soil_moisture_0_100_z", [0.0]*len(dates)), dtype=float)

    fig = plt.figure(figsize=(11, 6.2))
    ax1 = fig.add_subplot(2, 1, 1)
    ax1.plot(x, sm0, label="0-7cm")
    ax1.plot(x, sm28, label="7-28cm")
    ax1.plot(x, sm100, label="0-100cm")
    ax1.set_title("土壤含水量（多层）")
    ax1.set_ylabel("m³/m³")
    ax1.grid(True, alpha=0.2)
    ax1.legend(ncol=3, frameon=False)

    ax2 = fig.add_subplot(2, 1, 2, sharex=ax1)
    ax2.plot(x, z0, label="0-7cm 异常(z)")
    ax2.plot(x, z100, label="0-100cm 异常(z)")
    ax2.axhline(1.0, linestyle="--", linewidth=1.0, alpha=0.6)
    ax2.axhline(-1.0, linestyle="--", linewidth=1.0, alpha=0.6)
    ax2.set_title("土壤湿度异常（相对过去两周）")
    ax2.set_ylabel("z-score")
    ax2.grid(True, alpha=0.2)
    ax2.legend(frameon=False)

    step = max(1, len(dates)//10)
    ax2.set_xticks(x[::step])
    ax2.set_xticklabels([dates[i] for i in range(0, len(dates), step)], rotation=30, ha="right")
    ax2.set_xlabel("日期")
    _save(fig, out_png)

def plot_trigger_heatmap(dates: List[str], levels: Dict[str, List[str]], out_png: Path) -> None:
    _cn_font()
    hazards = ["waterlogging","drought","heat","frost","disease_pressure","wind_lodging"]
    level_map = {"L0":0,"L1":1,"L2":2,"L3":3,"":0,None:0}
    mat = []
    for h in hazards:
        arr = levels.get(h, [])
        mat.append([level_map.get(v,0) for v in arr])
    mat = np.array(mat, dtype=float)

    fig = plt.figure(figsize=(11, 3.6))
    ax = fig.add_subplot(1, 1, 1)
    im = ax.imshow(mat, aspect="auto", interpolation="nearest")
    ax.set_yticks(np.arange(len(hazards)))
    ax.set_yticklabels(hazards)
    step = max(1, len(dates)//12)
    ax.set_xticks(np.arange(0, len(dates), step))
    ax.set_xticklabels([dates[i] for i in range(0, len(dates), step)], rotation=30, ha="right")
    ax.set_title("风险等级日历（L0-L3）")
    cbar = fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02)
    cbar.set_ticks([0,1,2,3])
    cbar.set_ticklabels(["L0","L1","L2","L3"])
    _save(fig, out_png)

def generate_all_charts(dates: List[str], ind: Dict[str, List[float]], risk: Dict[str, List[float]], levels: Dict[str, List[str]], charts_dir: Path) -> Dict[str, str]:
    charts_dir.mkdir(parents=True, exist_ok=True)
    p1 = charts_dir / "risk_curves.png"
    p2 = charts_dir / "water_balance.png"
    p3 = charts_dir / "soil_moisture.png"
    p4 = charts_dir / "trigger_heatmap.png"
    plot_risk_curves(dates, risk, p1)
    plot_water_balance(dates, ind, p2)
    plot_soil_moisture(dates, ind, p3)
    plot_trigger_heatmap(dates, levels, p4)
    return {
        "risk_curves_png": str(p1).replace("\\","/"),
        "water_balance_png": str(p2).replace("\\","/"),
        "soil_moisture_png": str(p3).replace("\\","/"),
        "trigger_heatmap_png": str(p4).replace("\\","/"),
    }
