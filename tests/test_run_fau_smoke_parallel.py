"""Tests for the pure planning helpers of the parallel FAU smoke orchestrator."""

import pytest

from scripts.run_fau_smoke_parallel import plan_workers, read_question_lines


def test_plan_workers_one_distinct_gpu_per_episode():
    workers = plan_workers(8, [str(i) for i in range(8)], base_port=11500)
    assert len(workers) == 8
    # Each episode gets its own GPU and its own port/endpoint/output.
    assert [w["gpu_id"] for w in workers] == [str(i) for i in range(8)]
    assert [w["port"] for w in workers] == list(range(11500, 11508))
    assert workers[3]["endpoint"] == "http://127.0.0.1:11503"
    assert len({w["output_dir"] for w in workers}) == 8
    assert workers[2]["template_id"] == "fau_smoke_w2"


def test_plan_workers_round_robins_when_more_samples_than_gpus():
    workers = plan_workers(5, ["0", "1"], base_port=11500)
    assert [w["gpu_id"] for w in workers] == ["0", "1", "0", "1", "0"]
    # Ports stay unique even though GPUs repeat, so servers don't collide.
    assert len({w["port"] for w in workers}) == 5


def test_plan_workers_requires_gpus():
    with pytest.raises(ValueError):
        plan_workers(4, [])


def test_read_question_lines_takes_first_n(tmp_path):
    p = tmp_path / "q.jsonl"
    p.write_text('{"id": "a"}\n{"id": "b"}\n\n{"id": "c"}\n', encoding="utf-8")
    lines = read_question_lines(p, 2)
    assert lines == ['{"id": "a"}', '{"id": "b"}']
