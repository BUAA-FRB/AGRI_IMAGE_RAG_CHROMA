"""Smoke test (optional).
Run from project root:

python -m Report.tests_smoke --input advice_out/advice_output.json

"""

from __future__ import annotations

import argparse
from pathlib import Path
from .report_agent import ReportAgent, ReportAgentConfig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="advice_out/advice_output.json")
    ap.add_argument("--model_path", default="models/Qwen/Qwen2.5-3B-Instruct")
    args = ap.parse_args()

    assert Path(args.input).exists(), f"Missing input: {args.input}"
    assert Path(args.model_path).exists(), f"Missing model: {args.model_path}"

    agent = ReportAgent(ReportAgentConfig(model_path=args.model_path))
    md = agent.generate_report_markdown(args.input)
    print(md[:1200])
    print("\n... (truncated)\n")


if __name__ == "__main__":
    main()
