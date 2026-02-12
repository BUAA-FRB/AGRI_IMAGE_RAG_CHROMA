from __future__ import annotations

import argparse
from pathlib import Path

from .report_agent import ReportAgent, ReportAgentConfig


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="ReportAgent",
        description="Generate a high-quality Markdown report from advice_out/advice_output.json using Qwen2.5-3B-Instruct.",
    )
    p.add_argument(
        "--input",
        type=str,
        default="advice_out/advice_output.json",
        help="Path to advice output JSON (default: advice_out/advice_output.json)",
    )
    p.add_argument(
        "--model_path",
        type=str,
        default="models/Qwen/Qwen2.5-3B-Instruct",
        help="Local path to Qwen2.5-3B-Instruct (default: models/Qwen/Qwen2.5-3B-Instruct)",
    )
    p.add_argument(
        "--out_dir",
        type=str,
        default="report_out",
        help="Output directory for Markdown report (default: report_out)",
    )
    p.add_argument("--device", type=str, default="auto", help="auto|cpu|cuda|mps")
    p.add_argument("--dtype", type=str, default="auto", help="auto|float16|bfloat16|float32")
    p.add_argument("--max_new_tokens", type=int, default=1600)
    p.add_argument("--temperature", type=float, default=0.4)
    p.add_argument("--top_p", type=float, default=0.85)
    p.add_argument("--repetition_penalty", type=float, default=1.05)
    p.add_argument("--seed", type=int, default=42)
    return p


def main():
    args = build_parser().parse_args()

    in_path = Path(args.input)
    if not in_path.exists():
        raise SystemExit(f"Input JSON not found: {in_path.resolve()}")
    model_path = Path(args.model_path)
    if not model_path.exists():
        raise SystemExit(
            f"Model path not found: {model_path.resolve()}\n"
            "Please download/copy the model to the path or pass --model_path."
        )

    cfg = ReportAgentConfig(
        model_path=str(model_path),
        device=args.device,
        dtype=args.dtype,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        seed=args.seed,
    )
    agent = ReportAgent(cfg)
    out_path = agent.save_report(str(in_path), args.out_dir)
    print(out_path)


if __name__ == "__main__":
    main()
