"""Small I/O helpers: JSONL (plain or gzip) reading/writing, question loading."""
from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List


def open_text(path: str | Path, mode: str = "rt"):
    """Open a text file; ``.gz`` paths are compressed transparently."""
    path = Path(path)
    if path.suffix == ".gz":
        return gzip.open(path, mode, encoding="utf-8")
    return open(path, mode, encoding="utf-8")


def read_jsonl(path: str | Path, limit: int | None = None) -> Iterator[Dict[str, Any]]:
    n = 0
    with open_text(path) as fh:
        for line in fh:
            if not line.strip():
                continue
            yield json.loads(line)
            n += 1
            if limit and n >= limit:
                break


def load_jsonl(path: str | Path, limit: int | None = None) -> List[Dict[str, Any]]:
    return list(read_jsonl(path, limit))


def write_jsonl(path: str | Path, rows: Iterable[Dict[str, Any]]) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open_text(path, "wt") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    return n


def append_jsonl(path: str | Path, row: Dict[str, Any]) -> None:
    """Append one row and flush -- the unit of resumability for every long run."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
        fh.flush()


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _resolve(base: Path) -> Path:
    """Return ``base`` or its ``.gz`` twin, whichever exists (``.gz`` preferred)."""
    gz = base.with_suffix(base.suffix + ".gz")
    if gz.exists():
        return gz
    return base


def question_file(root: str | Path, dataset: str) -> Path:
    """Canonical location of a dataset's question file under ``data/questions``.

    Either ``<ds>_questions.jsonl`` or ``<ds>_questions.jsonl.gz`` is accepted; the
    shipped data is gzipped and every reader in this project handles both."""
    return _resolve(Path(root) / dataset / f"{dataset}_questions.jsonl")


def corpus_file(root: str | Path, dataset: str) -> Path:
    return _resolve(Path(root) / dataset / f"{dataset}_corpus.jsonl")
