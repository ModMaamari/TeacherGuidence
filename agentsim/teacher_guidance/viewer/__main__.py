"""Run the trajectory explorer: python -m agentsim.teacher_guidance.viewer"""

from __future__ import annotations

import argparse

from agentsim.teacher_guidance.viewer.server import serve


def main() -> None:
    parser = argparse.ArgumentParser(description="Teacher Guidance trajectory explorer")
    parser.add_argument(
        "--output-root",
        default="data/simulation_output",
        help="Directory containing run outputs (default: data/simulation_output)",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Bind host (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000, help="Bind port (default: 8000)")
    parser.add_argument("--no-browser", action="store_true", help="Do not open a browser")
    args = parser.parse_args()

    serve(
        output_root=args.output_root,
        host=args.host,
        port=args.port,
        open_browser=not args.no_browser,
    )


if __name__ == "__main__":
    main()
