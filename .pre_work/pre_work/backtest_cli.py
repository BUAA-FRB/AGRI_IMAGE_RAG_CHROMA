from __future__ import annotations
import argparse
from pathlib import Path
from typing import Optional, List
from .backtest import backtest_from_bundle

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Backtest for pre-work risk agent")
    p.add_argument("--bundle", type=str, required=True, help="path to bundle.json produced by run_agent")
    p.add_argument("--out_dir", type=str, default="", help="output dir; default: sibling of bundle")
    p.add_argument("--archive_model", type=str, default="era5_land")
    return p

def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    bundle = Path(args.bundle).resolve()
    out_dir = Path(args.out_dir).resolve() if args.out_dir else (bundle.parent / "backtest")
    out = backtest_from_bundle(bundle, out_dir, archive_model=str(args.archive_model))
    print(f"[OK] backtest -> {out.as_posix()}")
    return 0
