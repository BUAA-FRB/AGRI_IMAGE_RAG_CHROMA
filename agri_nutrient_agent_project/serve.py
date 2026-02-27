from __future__ import annotations

import argparse
import json
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles


def create_app(out_dir: Path) -> FastAPI:
    app = FastAPI(title="Nutrient Deficiency Agent Server")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # static files
    if not out_dir.exists():
        out_dir.mkdir(parents=True, exist_ok=True)

    app.mount("/nutrient_output", StaticFiles(directory=str(out_dir), html=False), name="nutrient_output")

    @app.get("/api/nutrient/latest")
    def get_latest():
        p = out_dir / "nutrient_deficiency_output.json"
        if not p.exists():
            raise HTTPException(status_code=404, detail=f"Missing output: {p}")
        return json.loads(p.read_text(encoding="utf-8"))

    return app


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="nutrient_output", help="Folder produced by run_agent.py")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", default=8000, type=int)
    args = ap.parse_args()

    import uvicorn
    app = create_app(Path(args.out_dir))
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
