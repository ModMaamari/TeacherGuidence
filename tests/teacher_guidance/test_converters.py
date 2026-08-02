"""Tests for the canonical multi-dataset converter package.

These are the cheap gate that has to catch a broken converter *before* an expensive
generation run: every converter is exercised on a realistic miniature of its source
format, the canonical schema is enforced, and each dataset's output is round-tripped
through the real local retriever to prove gold documents are actually reachable.
"""

from __future__ import annotations

import json

import pytest

from agentsim.teacher_guidance.converters import (
    CONVERTERS,
    ConversionError,
    convert_dataset,
    dataset_names,
    get_spec,
)
from agentsim.teacher_guidance.converters.base import (
    normalize_answer_text,
    split_sentences,
    validate_example,
)
from agentsim.teacher_guidance.converters.strategyqa import collect_evidence_ids
from agentsim.teacher_guidance.local_retrieval import HotpotLocalRetriever

# ---------------------------------------------------------------------------
# Miniature fixtures in each source's real on-disk shape
# ---------------------------------------------------------------------------
HOTPOT = {
    "id": "hp1",
    "question": "Where is the Oberoi Group head office?",
    "answer": "Delhi",
    "type": "bridge",
    "level": "medium",
    "supporting_facts": {"title": ["Oberoi family", "The Oberoi Group"], "sent_id": [0, 0]},
    "context": {
        "title": ["Oberoi family", "The Oberoi Group", "Distractor"],
        "sentences": [
            ["The Oberoi family runs hotels."],
            ["The Oberoi Group is headquartered in Delhi.", "It is large."],
            ["Unrelated text about penguins."],
        ],
    },
}

TWOWIKI = {
    "_id": "2w1",
    "type": "bridge_comparison",
    "question": "Which film has the director who died first?",
    "answer": "Coup de Torchon",
    "supporting_facts": [["Coup de Torchon", 0], ["Bertrand Tavernier", 1]],
    "evidences": [["Coup de Torchon", "director", "Bertrand Tavernier"]],
    "context": [
        ["Coup de Torchon", ["Coup de Torchon is a 1981 French film.", "It was acclaimed."]],
        ["Bertrand Tavernier", ["He was French.", "Tavernier died in 2021."]],
        ["Noise", ["Nothing relevant here."]],
    ],
}

MUSIQUE = {
    "id": "2hop__12345_6789",
    "question": "Who founded the company that makes the Pixel phone?",
    "answer": "Larry Page",
    "answer_aliases": ["Lawrence Page"],
    "answerable": True,
    "question_decomposition": [
        {"id": 1, "question": "Who makes Pixel?", "answer": "Google", "paragraph_support_idx": 0},
        {"id": 2, "question": "Who founded Google?", "answer": "Larry Page",
         "paragraph_support_idx": 1},
    ],
    "paragraphs": [
        {"idx": 0, "title": "Pixel", "is_supporting": True,
         "paragraph_text": "The Pixel is made by Google. It launched in 2016."},
        {"idx": 1, "title": "Google", "is_supporting": True,
         "paragraph_text": "Google was founded by Larry Page and Sergey Brin."},
        {"idx": 2, "title": "Penguins", "is_supporting": False,
         "paragraph_text": "Penguins are flightless birds. They live in cold regions."},
    ],
}

STRATEGYQA = {
    "qid": "sq1",
    "term": "Rock",
    "description": "a stone",
    "question": "Can a rock float on water?",
    "answer": False,
    "facts": ["Rocks are denser than water."],
    "decomposition": ["What is the density of rock?", "Is it greater than water?"],
    # annotator -> step -> evidence set -> paragraph ids (with sentinels mixed in)
    "evidence": [[[["p_rock"]], ["operation"]], [[["p_density"]], ["no_evidence"]]],
}

STRATEGYQA_PARAGRAPHS = {
    "p_rock": {"title": "Rock", "content": "Rock is a solid mineral material. It is dense."},
    "p_density": {"title": "Density", "content": "Density is mass per unit volume. Water is 1 g/cm3."},
    **{
        f"p_filler{i}": {"title": f"Filler {i}",
                         "content": f"Filler paragraph {i}. It has two sentences."}
        for i in range(20)
    },
}


def _convert_one(name):
    """Convert the fixture for ``name`` and return (questions, corpus, stats)."""
    if name == "hotpotqa":
        return convert_dataset("hotpotqa", [HOTPOT], "validation", strict=True)
    if name == "2wikimultihopqa":
        return convert_dataset("2wikimultihopqa", [TWOWIKI], "train", strict=True)
    if name == "musique":
        return convert_dataset("musique", [MUSIQUE], "train", strict=True)
    if name == "strategyqa":
        return convert_dataset(
            "strategyqa", [STRATEGYQA], "train", strict=True,
            paragraphs=STRATEGYQA_PARAGRAPHS,
            distractor_pool=sorted(STRATEGYQA_PARAGRAPHS),
            num_distractors=5, seed=13,
        )
    raise AssertionError(f"no fixture for {name}")


# ---------------------------------------------------------------------------
# Every registered dataset must convert and validate
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name", sorted(CONVERTERS))
def test_every_dataset_converts_and_validates(name):
    questions, corpus, stats = _convert_one(name)
    assert stats["converted"] == 1, f"{name}: {stats}"
    assert stats["skipped_total"] == 0
    # strict=True already validated, but assert explicitly so the contract is visible
    validate_example(questions[0], [d for d in corpus if d["qid"] == questions[0]["id"]])


@pytest.mark.parametrize("name", sorted(CONVERTERS))
def test_provenance_fields_are_stamped(name):
    questions, corpus, _ = _convert_one(name)
    spec = get_spec(name)
    q = questions[0]
    assert q["source"] == spec.source
    assert q["split"]
    assert q["gold_granularity"] == spec.gold_granularity
    assert q["answer_type"] == spec.answer_type
    for doc in corpus:
        assert doc["source"] == spec.source
        assert doc["split"] == q["split"]


@pytest.mark.parametrize("name", sorted(CONVERTERS))
def test_gold_granularity_governs_supporting_facts(name):
    """Paragraph-level sources must NOT invent sentence ids; sentence-level must have them."""
    questions, corpus, _ = _convert_one(name)
    q = questions[0]
    if q["gold_granularity"] == "paragraph":
        assert q["gold"]["supporting_facts"] == []
        assert all(d["gold_sent_ids"] == [] for d in corpus)
    else:
        assert q["gold"]["supporting_facts"]
        assert any(d["gold_sent_ids"] for d in corpus)


@pytest.mark.parametrize("name", sorted(CONVERTERS))
def test_gold_documents_are_reachable_by_the_real_retriever(tmp_path, name):
    """The end-to-end gate: gold docs must be retrievable for the question's own qid.

    A converter can produce schema-valid rows that the retriever still cannot serve (wrong
    qid grouping, empty text, doc_id mismatch). Running the real retriever here is what
    makes a new dataset trustworthy before any generation spend.
    """
    questions, corpus, _ = _convert_one(name)
    corpus_path = tmp_path / f"{name}_corpus.jsonl"
    corpus_path.write_text(
        "\n".join(json.dumps(r) for r in corpus) + "\n", encoding="utf-8"
    )
    retriever = HotpotLocalRetriever(str(corpus_path))
    q = questions[0]

    results = retriever.search(q["id"], q["query"], k=len(corpus))
    assert results, f"{name}: retriever returned nothing"
    returned = {r["doc_id"] for r in results}
    assert set(q["gold"]["gold_doc_ids"]) <= returned, (
        f"{name}: gold docs unreachable ({q['gold']['gold_doc_ids']} vs {sorted(returned)})"
    )
    for doc_id in q["gold"]["gold_doc_ids"]:
        doc = retriever.get_doc(doc_id)
        assert doc and doc["sentences"], f"{name}: {doc_id} has no usable text"


# ---------------------------------------------------------------------------
# Dataset-specific behaviour
# ---------------------------------------------------------------------------
def test_musique_carries_aliases_and_decomposition():
    questions, _, _ = _convert_one("musique")
    q = questions[0]
    assert q["gold"]["answer_aliases"] == ["Lawrence Page"]
    assert q["num_hops"] == 2
    assert q["type"] == "2hop"
    assert len(q["question_decomposition"]) == 2


def test_musique_skips_unanswerable():
    unanswerable = {**MUSIQUE, "id": "2hop__x_y", "answerable": False}
    questions, _, stats = convert_dataset("musique", [unanswerable], "train")
    assert questions == []
    assert stats["skipped_reasons"].get("unconvertible") == 1


def test_twowiki_keeps_evidence_triples_and_derives_hops():
    questions, _, _ = _convert_one("2wikimultihopqa")
    q = questions[0]
    assert q["evidence_triples"] == [["Coup de Torchon", "director", "Bertrand Tavernier"]]
    assert q["num_hops"] == 4  # bridge_comparison composes a bridge with a comparison


def test_strategyqa_boolean_answer_and_constructed_candidates():
    questions, corpus, _ = _convert_one("strategyqa")
    q = questions[0]
    assert q["answer"] == "no" and q["answer_type"] == "boolean"
    assert q["constructed_candidates"] is True
    # gold evidence retained, distractors added around it
    assert len(q["gold"]["gold_doc_ids"]) == 2
    assert len(corpus) == 7  # 2 gold + 5 distractors
    assert all(d.get("source_doc_id") for d in corpus)


def test_strategyqa_candidate_construction_is_deterministic():
    first, _, _ = _convert_one("strategyqa")
    second, _, _ = _convert_one("strategyqa")
    assert first[0]["retrieval_scope"]["candidate_doc_ids"] == \
        second[0]["retrieval_scope"]["candidate_doc_ids"]


def test_strategyqa_without_usable_evidence_is_skipped():
    no_evidence = {**STRATEGYQA, "qid": "sq2", "evidence": [[["operation"], ["no_evidence"]]]}
    questions, _, stats = convert_dataset(
        "strategyqa", [no_evidence], "train",
        paragraphs=STRATEGYQA_PARAGRAPHS, distractor_pool=sorted(STRATEGYQA_PARAGRAPHS),
    )
    assert questions == []
    assert stats["skipped_reasons"].get("unconvertible") == 1


def test_collect_evidence_ids_ignores_sentinels():
    ids = collect_evidence_ids([[[["a", "b"]], ["operation"]], [[["b", "c"]], ["no_evidence"]]])
    assert ids == ["a", "b", "c"]  # unique, order preserved, sentinels dropped


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("raw,expected,kind", [
    (True, "yes", "boolean"),
    (False, "no", "boolean"),
    ("TRUE", "yes", "boolean"),
    ("Yes", "yes", "boolean"),
    ("Delhi", "Delhi", "span"),
])
def test_normalize_answer_text(raw, expected, kind):
    assert normalize_answer_text(raw) == (expected, kind)


def test_split_sentences_handles_abbreviations_and_blank():
    assert split_sentences("") == []
    assert split_sentences("Dr. Who is a show. It aired in 1963.") == [
        "Dr. Who is a show.", "It aired in 1963."
    ]


# ---------------------------------------------------------------------------
# Validation actually rejects the mistakes it claims to
# ---------------------------------------------------------------------------
def _valid_pair():
    questions, corpus, _ = _convert_one("hotpotqa")
    return questions[0], corpus


@pytest.mark.parametrize("mutate,message", [
    (lambda q, c: q.update(answer=""), "empty answer"),
    (lambda q, c: q["gold"].update(gold_doc_ids=[]), "no gold documents"),
    (lambda q, c: q["gold"].update(gold_doc_ids=["hp1::doc99"]), "not in corpus"),
    (lambda q, c: q["retrieval_scope"].update(candidate_doc_ids=["hp1::doc0"]), "do not match"),
    (lambda q, c: q["gold"]["supporting_facts"].append({"title": "Nope", "sent_id": 0}),
     "not in corpus"),
    (lambda q, c: q["gold"]["supporting_facts"].append({"title": "Distractor", "sent_id": 99}),
     "out of sentence range"),
])
def test_validation_rejects_broken_examples(mutate, message):
    q, c = _valid_pair()
    mutate(q, c)
    with pytest.raises(ConversionError) as exc:
        validate_example(q, c)
    assert message in str(exc.value)


def test_convert_dataset_counts_skips_instead_of_crashing():
    broken = {"id": "bad", "question": "", "answer": "", "context": {}, "supporting_facts": {}}
    questions, _, stats = convert_dataset("hotpotqa", [HOTPOT, broken], "validation")
    assert len(questions) == 1
    assert stats["skipped_total"] == 1


def test_duplicate_qids_are_dropped():
    questions, _, stats = convert_dataset("hotpotqa", [HOTPOT, dict(HOTPOT)], "validation")
    assert len(questions) == 1
    assert stats["skipped_reasons"].get("duplicate_qid") == 1


def test_registry_is_complete():
    assert set(dataset_names()) == {"hotpotqa", "2wikimultihopqa", "musique", "strategyqa"}
    for name in dataset_names():
        spec = get_spec(name)
        assert spec.license and spec.homepage and spec.notes
