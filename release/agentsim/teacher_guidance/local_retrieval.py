"""
Local HotpotQA retrieval backend (``hotpot_local``).

Loads a corpus JSONL, groups documents by ``qid``, and ranks within a single
question's candidate documents. BM25 is used when ``rank_bm25`` is available; a
deterministic lexical-overlap scorer is used as a fallback so the backend works
without the optional dependency.

The retriever returns ranked previews to the student-visible layer but always keeps
full document text available via :meth:`get_doc` for the tool executor.
"""

from __future__ import annotations

import gzip
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

try:  # optional dependency
    from rank_bm25 import BM25Okapi

    _HAS_BM25 = True
except Exception:  # pragma: no cover - exercised only without the dep
    _HAS_BM25 = False


_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> List[str]:
    return _TOKEN_RE.findall((text or "").lower())


def _preview(text: str, max_chars: int = 240) -> str:
    text = (text or "").strip()
    return text if len(text) <= max_chars else text[:max_chars].rstrip() + "…"


class HotpotLocalRetriever:
    """Per-question lexical retriever over the HotpotQA corpus."""

    def __init__(self, corpus_path: str):
        self.corpus_path = str(corpus_path)
        self._docs_by_id: Dict[str, Dict[str, Any]] = {}
        self._doc_ids_by_qid: Dict[str, List[str]] = {}
        self._load()

    def _load(self) -> None:
        path = Path(self.corpus_path)
        if not path.exists():
            # accept either the plain or the gzipped form of the same corpus
            alt = path.with_suffix(path.suffix + ".gz") if path.suffix != ".gz" else path.with_suffix("")
            if alt.exists():
                path = alt
                self.corpus_path = str(alt)
            else:
                raise FileNotFoundError(f"Corpus not found: {path}")
        opener = gzip.open if path.suffix == ".gz" else open
        with opener(path, "rt", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                doc = json.loads(line)
                doc_id = doc["doc_id"]
                qid = doc.get("qid", "")
                self._docs_by_id[doc_id] = doc
                self._doc_ids_by_qid.setdefault(qid, []).append(doc_id)

    def get_doc(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """Return the full document dict (with text/sentences) or None."""
        return self._docs_by_id.get(doc_id)

    def _rank(self, query: str, docs: List[Dict[str, Any]], k: int) -> List[Dict[str, Any]]:
        if not docs:
            return []
        query_tokens = _tokenize(query)
        corpus_tokens = [_tokenize(d.get("text", "")) for d in docs]

        if _HAS_BM25 and query_tokens:
            bm25 = BM25Okapi(corpus_tokens)
            scores = bm25.get_scores(query_tokens)
        else:
            # Deterministic lexical-overlap fallback.
            query_set = set(query_tokens)
            scores = [
                (len(query_set & set(toks)) / (len(query_set) or 1))
                for toks in corpus_tokens
            ]

        ranked = sorted(
            zip(docs, scores), key=lambda pair: (pair[1], pair[0]["doc_id"]), reverse=True
        )
        results: List[Dict[str, Any]] = []
        for doc, score in ranked[: max(k, 0)]:
            results.append(
                {
                    "doc_id": doc["doc_id"],
                    "title": doc.get("title", ""),
                    "text_preview": _preview(doc.get("text", "")),
                    "score": round(float(score), 4),
                }
            )
        return results

    def search(
        self,
        qid: str,
        query: str,
        k: int = 5,
        candidate_doc_ids: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Rank documents for ``query`` within a single question's scope.

        Search is restricted to ``qid``; if ``candidate_doc_ids`` is given it is
        further restricted to that subset.
        """
        doc_ids = self._doc_ids_by_qid.get(qid, [])
        if candidate_doc_ids:
            allowed = set(candidate_doc_ids)
            doc_ids = [d for d in doc_ids if d in allowed]
        docs = [self._docs_by_id[d] for d in doc_ids if d in self._docs_by_id]
        return self._rank(query, docs, k)
