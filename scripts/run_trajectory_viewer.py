"""
Launch the Teacher Guidance trajectory explorer.

Usage:
    python scripts/run_trajectory_viewer.py --output-root data/simulation_output --port 8000
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agentsim.teacher_guidance.viewer.server import serve  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default="data/simulation_output")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()
    serve(args.output_root, args.host, args.port, open_browser=not args.no_browser)


if __name__ == "__main__":
    main()
