from __future__ import annotations
import sys
from pathlib import Path

THIS = Path(__file__).resolve()
ROOT = THIS.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pre_work.backtest_cli import main

if __name__ == "__main__":
    raise SystemExit(main())
