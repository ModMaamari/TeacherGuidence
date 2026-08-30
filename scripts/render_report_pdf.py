"""Render an Artifact-style HTML report fragment to a print-ready PDF.

Artifact pages are published as a body fragment (no <html>/<head>), so this wraps the
fragment in a minimal document, forces the light token set -- a PDF has no viewer theme --
and lets WeasyPrint apply the stylesheet's own @page/@media print rules.

Usage::

    python scripts/render_report_pdf.py reports/budget/budget_report.html out.pdf
"""
from __future__ import annotations
import sys
from pathlib import Path

from weasyprint import HTML

TEMPLATE = """<!DOCTYPE html>
<html lang="en" data-theme="light"><head><meta charset="utf-8">
<style>html{{color-scheme:light}} body{{margin:0}} img{{max-width:100%}}</style>
</head><body>{body}</body></html>"""


def main() -> int:
    src, dst = Path(sys.argv[1]), Path(sys.argv[2])
    html = TEMPLATE.format(body=src.read_text(encoding="utf-8"))
    HTML(string=html, base_url=str(src.parent.resolve())).write_pdf(str(dst))
    print(f"wrote {dst} ({dst.stat().st_size/1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
