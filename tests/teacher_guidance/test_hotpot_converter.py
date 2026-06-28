"""Tests for teacher_guidance.hotpot_converter."""

from agentsim.teacher_guidance.hotpot_converter import convert_example, convert_examples


HF_EXAMPLE = {
    "id": "q1",
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
            ["Unrelated text."],
        ],
    },
}

RAW_EXAMPLE = {
    "_id": "q2",
    "question": "Q?",
    "answer": "A",
    "type": "comparison",
    "level": "hard",
    "supporting_facts": [["Title A", 1]],
    "context": [
        ["Title A", ["s0", "s1 is gold"]],
        ["Title B", ["only distractor"]],
    ],
}


def test_convert_hf_example_marks_gold():
    q, corpus = convert_example(HF_EXAMPLE)
    assert q["id"] == "q1"
    assert q["gold"]["answer"] == "Delhi"
    assert set(q["gold"]["supporting_titles"]) == {"Oberoi family", "The Oberoi Group"}
    # candidate ids cover all three docs
    assert q["retrieval_scope"]["candidate_doc_ids"] == ["q1::doc0", "q1::doc1", "q1::doc2"]
    # gold docs are the two supporting titles
    assert q["gold"]["gold_doc_ids"] == ["q1::doc0", "q1::doc1"]

    by_id = {d["doc_id"]: d for d in corpus}
    assert by_id["q1::doc1"]["is_gold_doc"] is True
    assert by_id["q1::doc1"]["gold_sent_ids"] == [0]
    assert by_id["q1::doc2"]["is_gold_doc"] is False
    # text is the joined sentences
    assert by_id["q1::doc1"]["text"].startswith("The Oberoi Group is headquartered")


def test_convert_raw_example_shape():
    q, corpus = convert_example(RAW_EXAMPLE)
    assert q["id"] == "q2"
    assert q["gold"]["gold_doc_ids"] == ["q2::doc0"]
    by_id = {d["doc_id"]: d for d in corpus}
    assert by_id["q2::doc0"]["gold_sent_ids"] == [1]
    assert by_id["q2::doc1"]["is_gold_doc"] is False


def test_student_never_sees_gold_in_query_fields():
    q, _ = convert_example(HF_EXAMPLE)
    # The student-visible fields are query/type/level only; gold is nested separately.
    assert "Delhi" not in q["query"]
    assert q["gold"]["answer"] == "Delhi"


def test_convert_examples_aggregates():
    q_rows, c_rows = convert_examples([HF_EXAMPLE, RAW_EXAMPLE])
    assert len(q_rows) == 2
    assert len(c_rows) == 5  # 3 + 2
