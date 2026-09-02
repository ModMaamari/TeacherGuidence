"""Provenance stamps written onto every exported episode.

A public dataset has to answer three questions about any record years later: *what schema
is this*, *what code produced it*, and *under exactly what configuration*. Without those,
traces collected weeks apart silently become non-comparable -- which is precisely how a
release gets quietly corrupted.

``schema_version`` is bumped by hand whenever the episode record shape changes
incompatibly; ``framework_commit`` and ``config_hash`` are derived automatically.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional

#: Bump on any backward-incompatible change to the exported episode record.
EPISODE_SCHEMA_VERSION = "1.0"

_REPO_ROOT = Path(__file__).resolve().parents[2]


@lru_cache(maxsize=1)
def framework_commit() -> str:
    """Current git commit, suffixed ``+dirty`` when the tree has uncommitted changes.

    Returns ``"unknown"`` outside a git checkout (e.g. an installed wheel) rather than
    raising -- provenance is best-effort and must never break a generation run.
    """
    def _git(*args: str) -> Optional[str]:
        try:
            out = subprocess.run(
                ["git", *args], cwd=str(_REPO_ROOT),
                capture_output=True, text=True, timeout=10,
            )
        except Exception:  # noqa: BLE001 -- git may be missing entirely
            return None
        return out.stdout.strip() if out.returncode == 0 else None

    commit = _git("rev-parse", "HEAD")
    if not commit:
        return "unknown"
    dirty = _git("status", "--porcelain")
    return f"{commit}+dirty" if dirty else commit


def config_hash(mode_config: Dict[str, Any]) -> str:
    """Stable short hash of the generation configuration.

    Keys are sorted and the JSON is canonical, so the same configuration always hashes the
    same regardless of dict ordering. Runtime-only keys that do not change what the data
    *is* (paths, routing, concurrency) are excluded, so two runs that differ only in which
    provider served a call share a hash.
    """
    ignored = {
        "corpus_path", "teacher_router", "student_model", "teacher_model",
        "dataset", "dataset_split",
    }
    payload = {k: v for k, v in sorted((mode_config or {}).items()) if k not in ignored}
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]
