"""Tests for teacher_guidance.local_retrieval."""

import json

import pytest

from agentsim.teacher_guidance.local_retrieval import HotpotLocalRetriever


CORPUS = [
    {"doc_id": "q1::doc0", "qid": "q1", "title": "Oberoi family",
     "text": "The Oberoi family runs hotels in India.", "sentences": ["..."],
     "is_gold_doc": True, "gold_sent_ids": [0], "source": "hotpotqa", "split": "validation"},
    {"doc_id": "q1::doc1", "qid": "q1", "title": "The Oberoi Group",
     "text": "The Oberoi Group is headquartered in Delhi.", "sentences": ["..."],
     "is_gold_doc": True, "gold_sent_ids": [0], "source": "hotpotqa", "split": "validation"},
    {"doc_id": "q1::doc2", "qid": "q1", "title": "Distractor",
     "text": "A completely unrelated paragraph about penguins.", "sentences": ["..."],
     "is_gold_doc": False, "gold_sent_ids": [], "source": "hotpotqa", "split": "validation"},
    {"doc_id": "q2::doc0", "qid": "q2", "title": "Other question",
     "text": "Belongs to another question entirely.", "sentences": ["..."],
     "is_gold_doc": False, "gold_sent_ids": [], "source": "hotpotqa", "split": "validation"},
]


@pytest.fixture
def retriever(tmp_path):
    path = tmp_path / "corpus.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        for row in CORPUS:
            f.write(json.dumps(row) + "\n")
    return HotpotLocalRetriever(str(path))


def test_search_restricted_to_qid(retriever):
    results = retriever.search("q1", "Delhi headquarters", k=5)
    ids = {r["doc_id"] for r in results}
    assert ids <= {"q1::doc0", "q1::doc1", "q1::doc2"}
    assert "q2::doc0" not in ids


def test_search_ranks_relevant_first(retriever):
    results = retriever.search("q1", "headquartered in Delhi", k=3)
    assert results[0]["doc_id"] == "q1::doc1"


def test_candidate_doc_ids_respected(retriever):
    results = retriever.search("q1", "hotels", k=5, candidate_doc_ids=["q1::doc0"])
    assert [r["doc_id"] for r in results] == ["q1::doc0"]


def test_result_shape(retriever):
    results = retriever.search("q1", "hotels", k=1)
    r = results[0]
    assert set(r.keys()) == {"doc_id", "title", "text_preview", "score"}


def test_get_doc_returns_full_text(retriever):
    doc = retriever.get_doc("q1::doc1")
    assert doc is not None
    assert "Delhi" in doc["text"]
    assert retriever.get_doc("missing") is None
