# Advice/cli.py
import os
import sys
import glob
import argparse
import traceback

from .utils import read_json, write_json, ensure_dir, now_utc_iso
from .kb_indexer import build_kb_chroma
from .advice_engine import run_expert_advice
from .kb_retriever import Retriever


def _default_embed_model() -> str:
    local = os.path.abspath(os.path.join(".", "models", "bge-m3"))
    if os.path.isdir(local):
        return local
    return "BAAI/bge-m3"


def _default_label_map() -> str:
    p = os.path.abspath(os.path.join(".", "knowledge", "label_map.json"))
    return p if os.path.exists(p) else ""


def _default_qwen25_model_path() -> str:
    """
    你的真实本地路径：models\\Qwen\\Qwen2.5-3B-Instruct
    若存在则返回绝对路径，否则返回空字符串（由 run 时兜底报错）。
    """
    p = os.path.abspath(os.path.join(".", "models", "Qwen", "Qwen2.5-3B-Instruct"))
    return p if os.path.isdir(p) else ""


def _print_debug_banner(args) -> None:
    print("[debug] python =", sys.executable)
    print("[debug] cwd    =", os.path.abspath(os.getcwd()))
    print("[debug] cli    =", os.path.abspath(__file__))
    print("[debug] args   =", vars(args))


def cmd_kb_index(args):
    if getattr(args, "debug", False):
        _print_debug_banner(args)

    info = build_kb_chroma(
        kb_dir=args.kb_dir,
        db_dir=args.db_dir,
        collection=args.collection,
        embed_model=args.embed_model,
        max_chars=args.max_chars,
        overlap=args.overlap,
        batch_size=args.batch_size,
        reset=args.reset,
        kb_index_dir=args.kb_index_dir if args.kb_index_dir else None,
    )
    print("[kb-index] done:", info)


def cmd_kb_query(args):
    if getattr(args, "debug", False):
        _print_debug_banner(args)

    r = Retriever(db_dir=args.db_dir, collection=args.collection, embed_model=args.embed_model)
    hits = r.search(args.query, top_k=args.top_k, query_tags=args.tags or None)
    for i, h in enumerate(hits, start=1):
        meta = h.get("meta") or {}
        print(f"\n=== Hit {i} score={h.get('score'):.4f} dist={h.get('distance'):.4f} kid={h.get('kid')} ===")
        print(f"title={meta.get('title','')} section={meta.get('section','')} source={meta.get('source') or meta.get('path','')}")
        print((h.get("text") or "")[:900])


def cmd_run(args):
    if getattr(args, "debug", False):
        _print_debug_banner(args)

    # 1) 读 input
    fo = read_json(args.input_json)

    # 2) 准备输出目录
    out_dir = os.path.abspath(args.out_dir)
    ensure_dir(out_dir)

    # 3) 统一处理 qwen 路径：enable_llm 时必须是“本地存在的目录”
    qwen_path_used = None
    if args.enable_llm:
        qwen_path_used = (args.qwen25_model_path or "").strip() or _default_qwen25_model_path()
        qwen_path_used = os.path.abspath(qwen_path_used) if qwen_path_used else ""
        if not qwen_path_used or not os.path.isdir(qwen_path_used):
            raise FileNotFoundError(
                "enable_llm=True 但找不到本地 Qwen2.5 模型目录。\n"
                f"你传入的是: {args.qwen25_model_path!r}\n"
                f"默认尝试的是: {_default_qwen25_model_path()!r}\n"
                "请确认目录存在，例如：.\\models\\Qwen\\Qwen2.5-3B-Instruct"
            )

    # 4) 无论成功失败，都要落盘 advice_output.json
    status = "ok"
    err = None
    out = None

    try:
        out = run_expert_advice(
            final_output=fo,
            db_dir=args.db_dir,
            collection=args.collection,
            embed_model=args.embed_model,
            qwen25_model_path=qwen_path_used if args.enable_llm else None,
            kb_top_k=args.kb_top_k,
            enable_llm=args.enable_llm,
            label_map_path=args.label_map if args.label_map else None,
        )
    except Exception as e:
        status = "failed"
        err = str(e)
        tb = traceback.format_exc()
        print("[run] ERROR:", err)
        print(tb)

        out = {
            "schema": "advice.run_failed.v1",
            "error": err,
            "traceback": tb,
        }

    wrapped = {
        "schema": "advice.final_package.v3_chroma_norm",
        "time_utc": now_utc_iso(),
        "status": status,
        "error": err,
        "input_json": os.path.abspath(args.input_json),
        "db_dir": os.path.abspath(args.db_dir),
        "collection": args.collection,
        "embed_model": args.embed_model,
        "label_map": args.label_map or "",
        "qwen25_model_path": qwen_path_used or "",
        "output": out,
    }

    save_path = os.path.join(out_dir, "advice_output.json")
    write_json(save_path, wrapped)

    try:
        sz = os.path.getsize(save_path)
    except Exception:
        sz = -1
    print("[run] saved:", save_path, "bytes=", sz)


def cmd_batch(args):
    if getattr(args, "debug", False):
        _print_debug_banner(args)

    files = sorted(glob.glob(args.glob))
    if not files:
        print("[batch] no files matched:", args.glob)
        return

    for p in files:
        base = os.path.splitext(os.path.basename(p))[0]
        sub_out = os.path.join(os.path.abspath(args.out_dir), base)
        ns = argparse.Namespace(**vars(args))
        ns.input_json = p
        ns.out_dir = sub_out
        cmd_run(ns)


def main():
    ap = argparse.ArgumentParser(prog="Advice.cli")
    sub = ap.add_subparsers(dest="cmd", required=True)

    ap.add_argument("--debug", action="store_true", help="print debug info (cwd, cli path, args)")

    ap_i = sub.add_parser("kb-index", help="Build Chroma vector KB from knowledge folder")
    ap_i.add_argument("--kb_dir", required=True, help="knowledge directory (contains docs/ and source.jsonl etc.)")
    ap_i.add_argument("--db_dir", required=True, help="chroma persistent db dir")
    ap_i.add_argument("--collection", default="advice_kb")
    ap_i.add_argument("--embed_model", default=_default_embed_model(), help="local path recommended, e.g. .\\models\\bge-m3")
    ap_i.add_argument("--max_chars", type=int, default=900)
    ap_i.add_argument("--overlap", type=int, default=160)
    ap_i.add_argument("--batch_size", type=int, default=48)
    ap_i.add_argument("--reset", action="store_true")
    ap_i.add_argument("--kb_index_dir", default=os.path.abspath("./kb_index"), help="write readable index files here")
    ap_i.set_defaults(func=cmd_kb_index)

    ap_q = sub.add_parser("kb-query", help="Query Chroma KB for debugging retrieval")
    ap_q.add_argument("--db_dir", required=True)
    ap_q.add_argument("--collection", default="advice_kb")
    ap_q.add_argument("--embed_model", default=_default_embed_model())
    ap_q.add_argument("--query", required=True)
    ap_q.add_argument("--top_k", type=int, default=6)
    ap_q.add_argument("--tags", nargs="*", default=[])
    ap_q.set_defaults(func=cmd_kb_query)

    ap_r = sub.add_parser("run", help="Generate expert advice from final_output.v2 json using Chroma RAG")
    ap_r.add_argument("--input_json", required=True)
    ap_r.add_argument("--db_dir", required=True)
    ap_r.add_argument("--collection", default="advice_kb")
    ap_r.add_argument("--embed_model", default=_default_embed_model())
    ap_r.add_argument("--label_map", default=_default_label_map(), help="optional label alias map json, e.g. .\\knowledge\\label_map.json")
    ap_r.add_argument("--out_dir", required=True)
    ap_r.add_argument("--kb_top_k", type=int, default=6)
    ap_r.add_argument("--enable_llm", action="store_true")
    ap_r.add_argument(
        "--qwen25_model_path",
        default=_default_qwen25_model_path(),
        help="local Qwen2.5 model dir, e.g. .\\models\\Qwen\\Qwen2.5-3B-Instruct",
    )
    ap_r.set_defaults(func=cmd_run)

    ap_b = sub.add_parser("batch", help="Batch run over many final_output json files")
    ap_b.add_argument("--glob", required=True, help="e.g. .\\output\\*.json")
    ap_b.add_argument("--db_dir", required=True)
    ap_b.add_argument("--collection", default="advice_kb")
    ap_b.add_argument("--embed_model", default=_default_embed_model())
    ap_b.add_argument("--label_map", default=_default_label_map())
    ap_b.add_argument("--out_dir", required=True)
    ap_b.add_argument("--kb_top_k", type=int, default=6)
    ap_b.add_argument("--enable_llm", action="store_true")
    ap_b.add_argument(
        "--qwen25_model_path",
        default=_default_qwen25_model_path(),
        help="local Qwen2.5 model dir, e.g. .\\models\\Qwen\\Qwen2.5-3B-Instruct",
    )
    ap_b.set_defaults(func=cmd_batch)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
