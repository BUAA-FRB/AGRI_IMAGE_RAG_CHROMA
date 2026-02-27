from __future__ import annotations

import argparse
from pathlib import Path

from nutrient_agent.pipeline import run_nutrient_deficiency_agent


def main() -> None:
    ap = argparse.ArgumentParser(description="Run nutrient_deficiency downstream agent.")
    ap.add_argument("--input", default="output/latest.json", help="Upstream json path (agri_rag.final_output.v2).")
    ap.add_argument("--query", default=None, help="Query image path. Default: use upstream query_image_path; else Demo/query.jpg")
    ap.add_argument("--out_dir", default="nutrient_output", help="Output folder to create.")
    ap.add_argument("--force", action="store_true", help="Force run even if upstream doesn't predict nutrient_deficiency.")
    ap.add_argument("--llm", choices=["off", "qwen2.5", "qwen3-vl", "auto"], default="off",
                    help="Generate bilingual narrative using local Qwen models (optional). Default off.")
    ap.add_argument("--qwen25_path", default="models/Qwen/Qwen2.5-3B-Instruct", help="Local Qwen2.5-3B-Instruct path.")
    ap.add_argument("--qwen3vl_path", default="models/Qwen/Qwen3-VL-4B-Instruct", help="Local Qwen3-VL-4B-Instruct path.")
    ap.add_argument("--field_meta", default=None, help="Optional field meta json/geojson with bbox/center/polygon.")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    run_nutrient_deficiency_agent(
        upstream_path=Path(args.input),
        query_image_path=Path(args.query) if args.query else None,
        out_dir=out_dir,
        force=args.force,
        llm_mode=args.llm,
        qwen25_path=Path(args.qwen25_path),
        qwen3vl_path=Path(args.qwen3vl_path),
        field_meta_path=Path(args.field_meta) if args.field_meta else None,
    )


if __name__ == "__main__":
    main()
