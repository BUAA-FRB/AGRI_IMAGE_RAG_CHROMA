# agri_rag/logging_utils.py
from __future__ import annotations

import io
import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional


class _StreamToLogger(io.TextIOBase):
    """
    File-like stream that redirects writes to a logger.

    Compatibility:
    - transformers/tqdm may call sys.stdout.isatty() and access .encoding/.errors
    - some libs may call .writable(), .readable(), .fileno()
    - TextIOBase.encoding is read-only -> expose via @property (do NOT assign)
    """

    def __init__(
        self,
        logger: logging.Logger,
        level: int,
        fallback_stream: Optional[io.TextIOBase] = None,
    ) -> None:
        super().__init__()
        self.logger = logger
        self.level = level
        self._buffer = ""

        self._fallback = fallback_stream

        # store stream info in private vars (TextIOBase attributes are often read-only)
        self._encoding = getattr(fallback_stream, "encoding", "utf-8") if fallback_stream else "utf-8"
        self._errors = getattr(fallback_stream, "errors", "strict") if fallback_stream else "strict"
        self._newlines = getattr(fallback_stream, "newlines", None) if fallback_stream else None

    # ---- read-only properties expected by libs ----
    @property
    def encoding(self) -> str:  # type: ignore[override]
        return self._encoding

    @property
    def errors(self) -> str:  # type: ignore[override]
        return self._errors

    @property
    def newlines(self):  # type: ignore[override]
        return self._newlines

    # ---- core IO methods ----
    def write(self, msg: str) -> int:
        if msg is None:
            return 0
        if not isinstance(msg, str):
            msg = str(msg)
        if not msg:
            return 0

        self._buffer += msg

        # line-buffered logging to avoid per-char spam
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            line = line.rstrip("\r")
            if line.strip():
                self.logger.log(self.level, line)

        return len(msg)

    def flush(self) -> None:
        buf = self._buffer.strip()
        if buf:
            self.logger.log(self.level, buf)
        self._buffer = ""

        if self._fallback:
            try:
                self._fallback.flush()
            except Exception:
                pass

    # ---- compatibility helpers ----
    def isatty(self) -> bool:
        # transformers uses this for colored output decisions
        return False

    def readable(self) -> bool:
        return False

    def writable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return False

    def fileno(self) -> int:
        # some libs may query fileno; provide if fallback has it
        if self._fallback and hasattr(self._fallback, "fileno"):
            try:
                return int(self._fallback.fileno())  # type: ignore
            except Exception:
                pass
        raise io.UnsupportedOperation("fileno")

    def close(self) -> None:
        try:
            self.flush()
        finally:
            super().close()


def setup_logging(
    log_file: str,
    level: str = "INFO",
    to_console: bool = False,
    max_mb: int = 50,
    backup_count: int = 5,
    redirect_std: bool = True,
) -> None:
    """
    Configure root logging:
      - Rotating file handler (always)
      - Optional console handler
      - Optionally redirect stdout/stderr to logging

    Recommended for your requirement "everything to logs":
      to_console=False, redirect_std=True
    """
    # Reduce noisy progress bars from HF downloads
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    lvl = getattr(logging, level.upper(), logging.INFO)

    log_path = Path(log_file).resolve()
    log_path.parent.mkdir(parents=True, exist_ok=True)

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    handlers: list[logging.Handler] = []

    fh = RotatingFileHandler(
        filename=str(log_path),
        maxBytes=int(max_mb) * 1024 * 1024,
        backupCount=int(backup_count),
        encoding="utf-8",
    )
    fh.setLevel(lvl)
    fh.setFormatter(fmt)
    handlers.append(fh)

    if to_console:
        # IMPORTANT: use original stderr to avoid recursion when std is redirected
        sh = logging.StreamHandler(stream=sys.__stderr__)
        sh.setLevel(lvl)
        sh.setFormatter(fmt)
        handlers.append(sh)

    root = logging.getLogger()
    root.setLevel(lvl)
    root.handlers.clear()
    for h in handlers:
        root.addHandler(h)

    logging.captureWarnings(True)

    # Redirect std streams only when console handler is OFF (avoid recursion)
    if redirect_std and not to_console:
        std_logger = logging.getLogger("agri_rag.std")
        sys.stdout = _StreamToLogger(std_logger, logging.INFO, fallback_stream=sys.__stdout__)   # type: ignore
        sys.stderr = _StreamToLogger(std_logger, logging.ERROR, fallback_stream=sys.__stderr__)  # type: ignore

        # Extra guard: some libs may probe isatty attribute existence
        if not hasattr(sys.stdout, "isatty"):
            sys.stdout.isatty = lambda: False  # type: ignore
        if not hasattr(sys.stderr, "isatty"):
            sys.stderr.isatty = lambda: False  # type: ignore

    # Reduce some third-party logger verbosity
    for noisy in ["urllib3", "PIL", "matplotlib", "transformers", "datasets", "chromadb"]:
        logging.getLogger(noisy).setLevel(max(lvl, logging.WARNING))
