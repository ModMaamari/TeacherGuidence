"""tgd -- teacher-guidance distillation toolkit (library code shared by scripts/).

Importing this package puts the project root on ``sys.path`` so ``agentsim`` (the
simulation harness) resolves regardless of the working directory.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DATASETS = ["hotpotqa", "2wikimultihopqa", "musique", "strategyqa"]
