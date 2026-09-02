"""Shared logging + artifact helpers.

Every script uses ``setup_logger`` (UTC-timestamped console + file logging) and
``write_json`` (atomic JSON artifacts stamped with their write time).
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict


def utc_ts() -> str:
    """Filesystem-safe UTC timestamp, second resolution: 20260711T142530Z."""
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def timestamped_dir(base: str | Path, tag: str = "") -> Path:
    """Create and return ``<base>/<utc_ts>[_<tag>]/`` (parents included)."""
    name = utc_ts() + (f"_{tag}" if tag else "")
    path = Path(base) / name
    path.mkdir(parents=True, exist_ok=True)
    return path


def setup_logger(name: str, log_file: str | Path | None = None) -> logging.Logger:
    """Logger with UTC timestamps on stderr and (optionally) a file."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if logger.handlers:  # already configured (re-entry in same process)
        return logger
    fmt = logging.Formatter(
        "%(asctime)s.%(msecs)03dZ | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    fmt.converter = time.gmtime
    sh = logging.StreamHandler(sys.stderr)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    if log_file is not None:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file)
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger


def write_json(path: str | Path, obj: Dict[str, Any]) -> None:
    """Atomic timestamp-stamped JSON artifact writer."""
    obj = dict(obj)
    obj.setdefault("written_at_utc", datetime.now(timezone.utc).isoformat())
    tmp = str(path) + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)
