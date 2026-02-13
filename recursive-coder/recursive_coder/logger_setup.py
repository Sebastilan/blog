"""Structured logging setup: console (concise) + file (full detail)."""

from __future__ import annotations

import logging
import sys
from pathlib import Path


class _ConsoleFormatter(logging.Formatter):
    """Compact colored formatter for terminal output."""

    COLORS = {
        logging.DEBUG: "\033[90m",     # grey
        logging.INFO: "\033[36m",      # cyan
        logging.WARNING: "\033[33m",   # yellow
        logging.ERROR: "\033[31m",     # red
        logging.CRITICAL: "\033[1;31m",# bold red
    }
    RESET = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        color = self.COLORS.get(record.levelno, "")
        ts = self.formatTime(record, "%H:%M:%S")
        task_id = getattr(record, "task_id", "")
        tag = f" [{task_id}]" if task_id else ""
        return f"{color}[{ts}] [{record.levelname[0]}]{tag} {record.getMessage()}{self.RESET}"


class _FileFormatter(logging.Formatter):
    """Verbose formatter for log files."""

    def format(self, record: logging.LogRecord) -> str:
        ts = self.formatTime(record, "%Y-%m-%d %H:%M:%S")
        task_id = getattr(record, "task_id", "-")
        module = record.module
        return f"[{ts}] [{record.levelname}] [{module}] [{task_id}] {record.getMessage()}"


def setup_logging(workspace_dir: str | None = None, verbose: bool = False) -> None:
    """Configure root logger with console + optional file output."""
    root = logging.getLogger("recursive_coder")
    root.setLevel(logging.DEBUG)
    root.handlers.clear()

    # Console handler
    console = logging.StreamHandler(sys.stderr)
    console.setLevel(logging.DEBUG if verbose else logging.INFO)
    console.setFormatter(_ConsoleFormatter())
    root.addHandler(console)

    # File handler (if workspace provided)
    if workspace_dir:
        log_path = Path(workspace_dir) / "run.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(str(log_path), encoding="utf-8")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(_FileFormatter())
        root.addHandler(fh)


def get_logger(name: str) -> logging.Logger:
    """Get a child logger under the recursive_coder namespace."""
    return logging.getLogger(f"recursive_coder.{name}")
