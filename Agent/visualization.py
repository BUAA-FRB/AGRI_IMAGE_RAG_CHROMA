import os
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib import font_manager

from .utils import ensure_dir, safe_relpath
from .io import resolve_asset_path, resolve_assets_root


# ----------------------------
# Font setup (Chinese-friendly)
# ----------------------------

_CN_FONT_INITIALIZED = False


def _setup_cn_font() -> None:
    """
    Make matplotlib render Chinese correctly (common issue on Windows).
    Strategy:
    - Try a list of common Chinese fonts and use the first available.
    - If none found, leave default; chart may still show squares.
    """
    global _CN_FONT_INITIALIZED
    if _CN_FONT_INITIALIZED:
        return

    # Common Chinese fonts across Windows/macOS/Linux
    candidates = [
        # Windows
        "Microsoft YaHei",
        "SimHei",
        "SimSun",
        "NSimSun",
        # macOS
        "PingFang SC",
        "Heiti SC",
        "Songti SC",
        # Linux (common)
        "Noto Sans CJK SC",
        "Noto Sans CJK",
        "WenQuanYi Micro Hei",
        "WenQuanYi Zen Hei",
        "Source Han Sans SC",
        "Source Han Sans CN",
    ]

    try:
        available = {f.name for f in font_manager.fontManager.ttflist}
    except Exception:
        available = set()

    chosen = None
    for name in candidates:
        if name in available:
            chosen = name
            break

    if chosen:
        plt.rcParams["font.sans-serif"] = [chosen]
        plt.rcParams["axes.unicode_minus"] = False  # avoid minus sign being rendered as a box
    # else: do nothing, fallback to default

    _CN_FONT_INITIALIZED = True


# ----------------------------
# Timeline
# ----------------------------

def visualize_timeline(strategy: Dict[str, Any], out_png: str) -> None:
    _setup_cn_font()

    phases = ["灾前", "灾中", "灾后"]
    counts = [len(strategy.get("phases", {}).get(p, [])) for p in phases]

    plt.figure(figsize=(8, 3))
    plt.barh(phases, counts)
    plt.xlabel("行动模块数量")

    s = strategy.get("summary", {})
    title = f"策略时间线 | {s.get('playbook_key','')} | 严重度: {s.get('severity','')}"
    plt.title(title)

    # Make it look nicer when counts are small integers
    try:
        xmax = max(counts) if counts else 0
        plt.xlim(0, max(1, xmax) + 0.2)
    except Exception:
        pass

    plt.tight_layout()
    ensure_dir(os.path.dirname(out_png) or ".")
    plt.savefig(out_png, dpi=200)
    plt.close()


# ----------------------------
# Evidence grid
# ----------------------------

def _read_image(path: str):
    try:
        return mpimg.imread(path)
    except Exception:
        return None


def build_evidence_grid(
    final_assets: Dict[str, Any],
    evidence_order: List[str],
    out_png: str,
    max_evidence: int = 6
) -> None:
    """
    使用 assets.query.preview_path + assets.evidence[Ei].preview_path 拼图
    """
    _setup_cn_font()

    # 解析 assets 根目录
    dummy = {"assets": final_assets, "src_json_path": ""}  # 只为了复用 resolve_asset_path 形式
    class _Tmp: pass
    fo = _Tmp()
    fo.assets = final_assets
    fo.src_json_path = final_assets.get("output_dir", "") or ""
    _ = resolve_assets_root(fo)  # type: ignore  # kept for compatibility, root may be useful later

    q_prev = final_assets.get("query", {}).get("preview_path")
    q_path = resolve_asset_path(fo, q_prev)  # type: ignore
    ev_map = final_assets.get("evidence", {}) or {}

    # 取前 max_evidence
    eids = [e for e in evidence_order if e in ev_map][:max_evidence]

    # grid: 1(query) + k evidence
    imgs: List[Tuple[str, Optional[str]]] = [("Query", q_path)]
    for eid in eids:
        imgs.append((eid, resolve_asset_path(fo, ev_map[eid].get("preview_path"))))  # type: ignore

    n = len(imgs)
    cols = 3
    rows = (n + cols - 1) // cols

    plt.figure(figsize=(12, 4 * rows))
    for i, (title, path) in enumerate(imgs):
        ax = plt.subplot(rows, cols, i + 1)
        ax.axis("off")
        ax.set_title(title)
        if path and os.path.exists(path):
            im = _read_image(path)
            if im is not None:
                ax.imshow(im)
            else:
                ax.text(0.1, 0.5, f"无法读取图片\n{path}", fontsize=10)
        else:
            ax.text(0.1, 0.5, f"缺失图片\n{path}", fontsize=10)

    plt.tight_layout()
    ensure_dir(os.path.dirname(out_png) or ".")
    plt.savefig(out_png, dpi=200)
    plt.close()


# ----------------------------
# Dashboard HTML
# ----------------------------

def write_dashboard_html(
    out_html: str,
    base_dir: str,
    title: str,
    timeline_png: str,
    evidence_grid_png: str,
    montage_png: Optional[str],
    strategy: Dict[str, Any],
    final_output: Dict[str, Any],
    refine_json: Optional[Dict[str, Any]] = None,
) -> None:
    ensure_dir(os.path.dirname(out_html) or ".")
    tl_rel = safe_relpath(timeline_png, base_dir)
    grid_rel = safe_relpath(evidence_grid_png, base_dir)
    montage_rel = safe_relpath(montage_png, base_dir) if montage_png else None

    s = strategy.get("summary", {})
    mf = strategy.get("model_fields", {})
    key_takeaway = ""
    if refine_json:
        key_takeaway = str(refine_json.get("key_takeaway") or "")

    html = f"""<!doctype html>
<html lang="zh">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>{title}</title>
<style>
body {{ font-family: Arial, "Microsoft YaHei", sans-serif; margin: 24px; }}
.grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 18px; }}
.card {{ border: 1px solid #ddd; border-radius: 10px; padding: 16px; }}
img {{ max-width: 100%; height: auto; border-radius: 8px; }}
h2 {{ margin-top: 0; }}
code, pre {{ background: #f6f8fa; padding: 10px; border-radius: 8px; overflow-x: auto; }}
.small {{ color: #666; font-size: 12px; }}
</style>
</head>
<body>
<h1>{title}</h1>
<div class="small">schema={final_output.get("schema","")} | id={final_output.get("id","")} | time={final_output.get("time_utc","")}</div>

<div class="grid">
  <div class="card">
    <h2>预测摘要</h2>
    <ul>
      <li>label: <b>{s.get("pred_label","")}</b></li>
      <li>playbook: <b>{s.get("playbook_key","")}</b></li>
      <li>confidence: <b>{s.get("confidence",0):.2f}</b></li>
      <li>severity: <b>{s.get("severity","")}</b></li>
      <li>evidence refs: {", ".join(s.get("evidence_refs",[]) or [])}</li>
    </ul>
    <h3>关键结论（可选）</h3>
    <div>{key_takeaway if key_takeaway else "（未启用 Qwen2.5-3B 整理）"}</div>
    <h3>模型结论</h3>
    <div>{mf.get("conclusion","")}</div>
    <h3>不确定性</h3>
    <div>{mf.get("uncertainty","")}</div>
  </div>

  <div class="card">
    <h2>策略时间线</h2>
    <img src="{tl_rel}" alt="timeline"/>
  </div>

  <div class="card">
    <h2>证据图册（Query + Top Evidence）</h2>
    <img src="{grid_rel}" alt="evidence grid"/>
  </div>

  <div class="card">
    <h2>原始拼图（来自你的 output assets）</h2>
    {f'<img src="{montage_rel}" alt="montage"/>' if montage_rel else "<div>（未提供 montage 路径）</div>"}
  </div>
</div>

<div class="card" style="margin-top:18px;">
  <h2>灾前 / 灾中 / 灾后行动清单（结构化）</h2>
  <pre>{strategy}</pre>
</div>

</body></html>
"""
    with open(out_html, "w", encoding="utf-8") as f:
        f.write(html)
