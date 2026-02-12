from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

from .config import SiteConfig, RunConfig
from .pipeline import run_site
from .qwen_runner import QwenRunner, QwenGenConfig

class _LLMAdapter:
    def __init__(self, model_path: str, device: str, dtype: str, gen_cfg: QwenGenConfig) -> None:
        self.runner = QwenRunner(model_path=model_path, device=device, dtype=dtype)
        self.gen_cfg = gen_cfg

    def generate(self, messages: List[Dict[str, str]]) -> str:
        return self.runner.generate(messages, self.gen_cfg)

def _load_sites(path: str) -> List[SiteConfig]:
    obj = json.loads(Path(path).read_text(encoding="utf-8"))
    sites = obj.get("sites") if isinstance(obj, dict) else None
    if not isinstance(sites, list):
        raise ValueError("sites.json must be like {\"sites\": [...]}")
    out: List[SiteConfig] = []
    for s in sites:
        if not isinstance(s, dict):
            continue
        out.append(SiteConfig(
            name=str(s.get("name")),
            lat=float(s.get("lat")),
            lon=float(s.get("lon")),
            crop=str(s.get("crop","unknown")),
            stage=str(s.get("stage","unknown")),
            timezone=str(s.get("timezone","auto")),
        ))
    return out

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Pre-work risk warning agent (weather-hydro-soil)")
    p.add_argument("--sites", type=str, default="", help="JSON file: {sites:[{name,lat,lon,crop,stage}]}")
    p.add_argument("--name", type=str, default="", help="single site name")
    p.add_argument("--lat", type=float, default=None)
    p.add_argument("--lon", type=float, default=None)
    p.add_argument("--crop", type=str, default="unknown")
    p.add_argument("--stage", type=str, default="unknown")

    p.add_argument("--days_hist", type=int, default=14)
    p.add_argument("--days_fore", type=int, default=16)
    p.add_argument("--extend_to_30", action="store_true")

    p.add_argument("--no_soilgrids", action="store_true")
    p.add_argument("--out_root", type=str, default=".pre_work_out")

    p.add_argument("--no_llm", action="store_true")
    p.add_argument("--model_path", type=str, default="./models/Qwen/Qwen2.5-3B-Instruct")
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--dtype", type=str, default="auto")
    return p

def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    rc = RunConfig(
        days_hist=int(args.days_hist),
        days_fore=int(args.days_fore),
        extend_to_30=bool(args.extend_to_30),
        use_soilgrids=not bool(args.no_soilgrids),
        out_root=str(args.out_root),
        use_llm=not bool(args.no_llm),
        model_path=str(args.model_path),
        device=str(args.device),
        dtype=str(args.dtype),
    )

    llm = None
    if rc.use_llm:
        gen_cfg = QwenGenConfig(
            max_new_tokens=rc.max_new_tokens,
            temperature=rc.temperature,
            top_p=rc.top_p,
            repetition_penalty=rc.repetition_penalty,
            seed=rc.seed,
        )
        try:
            llm = _LLMAdapter(rc.model_path, rc.device, rc.dtype, gen_cfg)
        except Exception as e:
            print(f"[WARN] failed to init LLM: {type(e).__name__}: {e}")
            llm = None

    if args.sites:
        sites = _load_sites(args.sites)
    else:
        if not args.name or args.lat is None or args.lon is None:
            raise SystemExit("Provide --sites OR ( --name --lat --lon )")
        sites = [SiteConfig(name=args.name, lat=float(args.lat), lon=float(args.lon), crop=args.crop, stage=args.stage)]

    for s in sites:
        out_dir = run_site(s, rc, llm_runner=llm)
        print(f"[OK] {s.name} -> {out_dir.as_posix()}")

    return 0
