import argparse
import glob
import os

from .agent_runtime import AgriAgent
from .utils import ensure_dir, write_json


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("agri_rag.Agent (json-driven)")

    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("run", help="从 final_output.v2 json 生成灾前/灾中/灾后策略 + 可视化 + 报告")
    sp.add_argument("--input_json", required=True, help="你的 output/*.json（agri_rag.final_output.v2）")
    sp.add_argument("--out_dir", default="./agent_runs", help="输出目录（自动创建 run_id 子目录）")
    sp.add_argument("--notes", default="", help="额外说明（写入策略摘要）")
    sp.add_argument("--max_evidence", type=int, default=6, help="证据图册最多展示多少条 evidence")
    sp.add_argument("--enable_qwen25_refine", action="store_true", help="启用 Qwen2.5-3B 策略整理（可选）")
    sp.add_argument("--qwen25_model_path", default="./models/Qwen2.5-3B", help="本地 Qwen2.5-3B 路径")
    sp.add_argument("--save_json", default="", help="额外保存一份 agent_output.json 到指定路径")

    sp2 = sub.add_parser("batch", help="批量处理某个目录下的 final json")
    sp2.add_argument("--glob", default="./output/*.json", help="glob pattern，例如 ./output/*.json")
    sp2.add_argument("--out_dir", default="./agent_runs", help="输出目录")
    sp2.add_argument("--enable_qwen25_refine", action="store_true")
    sp2.add_argument("--qwen25_model_path", default="./models/Qwen2.5-3B")

    return p


def main() -> None:
    args = build_parser().parse_args()

    if args.cmd == "run":
        agent = AgriAgent(
            enable_qwen25_refine=bool(args.enable_qwen25_refine),
            qwen25_model_path=args.qwen25_model_path,
        )
        ensure_dir(args.out_dir)
        out = agent.run_from_json(
            final_json_path=args.input_json,
            out_dir=args.out_dir,
            notes=args.notes,
            max_evidence=int(args.max_evidence),
        )
        if args.save_json:
            ensure_dir(os.path.dirname(args.save_json) or ".")
            write_json(args.save_json, out)

        print("[Agent] done.")
        v = out.get("visuals", {})
        print(" - report_md:", v.get("report_md"))
        print(" - dashboard_html:", v.get("dashboard_html"))
        print(" - timeline_png:", v.get("timeline_png"))
        print(" - evidence_grid_png:", v.get("evidence_grid_png"))

    elif args.cmd == "batch":
        files = sorted(glob.glob(args.glob))
        if not files:
            print(f"[Agent] no files matched: {args.glob}")
            return
        agent = AgriAgent(
            enable_qwen25_refine=bool(args.enable_qwen25_refine),
            qwen25_model_path=args.qwen25_model_path,
        )
        ensure_dir(args.out_dir)
        for fp in files:
            try:
                out = agent.run_from_json(fp, out_dir=args.out_dir)
                print(f"[OK] {fp} -> {out.get('run_id')}")
            except Exception as e:
                print(f"[FAIL] {fp}: {e}")


if __name__ == "__main__":
    main()
