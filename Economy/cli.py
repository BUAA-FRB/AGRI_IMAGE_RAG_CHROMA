from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

from .economy_agent import EconomyAgent, EconomyConfig
from .io_utils import read_json, write_json, write_text
from .charts import generate_all_charts
from .prompt_builder import build_llm_prompt
from .llm_qwen import LocalQwenClient, QwenGenConfig
from .report import build_economy_report_markdown


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("Economy Loss Estimator + Local Qwen Analysis + Multimodal Report")
    p.add_argument("--latest", type=str, default="output/latest.json", help="Path to output/latest.json")
    p.add_argument("--field-stats", type=str, default="datasets/data2018_miniscale/field_stats.json", help="Path to field_stats.json")
    p.add_argument("--out-dir", type=str, default="./economy_out", help="Output directory (json/md/prompt/llm)")
    p.add_argument("--params-dir", type=str, default="Economy/params", help="Economy params directory")

    p.add_argument("--crop", type=str, default="corn", choices=["corn", "rice", "wheat"], help="Crop type")
    p.add_argument("--stage", type=str, default="generic", help="Growth stage (placeholder)")

    p.add_argument("--area-mode", type=str, default="gsd", choices=["gsd", "ratio_only"], help="Area conversion mode")
    p.add_argument("--gsd-m-per-px", type=float, default=0.1, help="GSD meters/pixel when area-mode=gsd")
    p.add_argument("--normalize-to", type=str, default="mu", choices=["mu", "ha"], help="If ratio_only, normalize money per mu/ha")

    # charts
    p.add_argument("--forecast-months", type=int, default=6, help="Forecast horizon in months for charts")

    # llm
    p.add_argument("--no-llm", action="store_true", help="Disable local LLM analysis")
    p.add_argument("--llm-model-dir", type=str, default="./models/Qwen/Qwe2.5-3B-Instruct", help="Local Qwen model directory")
    p.add_argument("--max-new-tokens", type=int, default=1200)
    p.add_argument("--temperature", type=float, default=0.4)
    p.add_argument("--top-p", type=float, default=0.9)
    p.add_argument("--repetition-penalty", type=float, default=1.05)
    return p


def main() -> None:
    args = build_parser().parse_args()

    latest_path = Path(args.latest)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    latest = read_json(latest_path)

    # 1) rule-based economy estimate
    agent = EconomyAgent(params_dir=args.params_dir)
    cfg = EconomyConfig(
        crop=args.crop,
        stage=args.stage,
        area_mode=args.area_mode,
        gsd_m_per_px=args.gsd_m_per_px if args.area_mode == "gsd" else None,
        normalize_to=args.normalize_to,
    )

    econ = agent.estimate(
        latest_json_path=args.latest,
        field_stats_path=args.field_stats,
        cfg=cfg,
    )

    # write structured output
    economy_output_path = out_dir / "economy_output.json"
    write_json(economy_output_path, econ.to_dict())

    # 2) charts directory under output/assets/charts/<runid>_<timestamp>/
    run_id = econ.run_id
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    # decide output root directory from latest.json assets.output_dir if exists else parent of latest.json
    output_root = Path(latest.get("assets", {}).get("output_dir") or latest_path.parent)
    charts_dir = output_root / "assets" / "charts" / f"{run_id}_{ts}"
    charts_meta = generate_all_charts(econ, charts_dir=charts_dir, n_months=int(args.forecast_months))

    # 3) build prompt and call local Qwen
    prompt = build_llm_prompt(latest=latest, econ=econ, charts_meta=charts_meta)
    prompt_path = out_dir / "economy_prompt.txt"
    write_text(prompt_path, f"[SYSTEM]\n{prompt['system']}\n\n[USER]\n{prompt['user']}\n")

    llm_md = ""
    llm_out_path = out_dir / "llm_analysis.md"

    if not args.no_llm:
        try:
            client = LocalQwenClient(model_dir=args.llm_model_dir)
            gen_cfg = QwenGenConfig(
                max_new_tokens=int(args.max_new_tokens),
                temperature=float(args.temperature),
                top_p=float(args.top_p),
                repetition_penalty=float(args.repetition_penalty),
            )
            llm_md = client.generate_markdown(prompt, gen_cfg=gen_cfg)
            write_text(llm_out_path, llm_md)
        except Exception as e:
            llm_md = f"> 大模型调用失败：{type(e).__name__}: {e}\n\n> 你可以先用 `--no-llm` 生成规则报告与图表，或检查 transformers/torch/模型路径。"
            write_text(llm_out_path, llm_md)
    else:
        write_text(llm_out_path, "> 未启用大模型（--no-llm）。")

    # 4) final multimodal report
    report_md = build_economy_report_markdown(
        latest=latest,
        econ=econ,
        charts_meta=charts_meta,
        out_dir=out_dir,
        llm_md=llm_md,
    )
    report_path = out_dir / "economy_report.md"
    write_text(report_path, report_md)

    print(f"[Economy] Wrote: {economy_output_path}")
    print(f"[Economy] Wrote: {prompt_path}")
    print(f"[Economy] Wrote: {llm_out_path}")
    print(f"[Economy] Wrote: {report_path}")
    print(f"[Economy] Charts: {charts_dir}")


if __name__ == "__main__":
    main()
