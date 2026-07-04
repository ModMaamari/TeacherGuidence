"""Tests for the sharded-run consolidation helpers."""

from pathlib import Path

from scripts.consolidate_run_shards import find_sample_dirs, plan_moves, consolidate


def _make_episode(dir_path: Path):
    dir_path.mkdir(parents=True, exist_ok=True)
    (dir_path / "teacher_guidance_episodes.jsonl").write_text('{"qid": "x"}\n', encoding="utf-8")


def test_find_sample_dirs_across_shards(tmp_path):
    for w in range(3):
        for s in range(2):
            _make_episode(tmp_path / f"w{w}" / "uuid" / "hotpot_questions" / f"sample_{s:03d}")
    dirs = find_sample_dirs(tmp_path, exclude_name="fau_run100")
    assert len(dirs) == 6
    # Deterministic order.
    assert dirs == sorted(dirs, key=lambda p: str(p))


def test_find_sample_dirs_excludes_target_run(tmp_path):
    _make_episode(tmp_path / "w0" / "uuid" / "hotpot_questions" / "sample_001")
    _make_episode(tmp_path / "combined" / "hotpot_questions" / "sample_0001")
    dirs = find_sample_dirs(tmp_path, exclude_name="combined")
    assert len(dirs) == 1
    assert "combined" not in dirs[0].parts


def test_plan_moves_renumbers_sequentially(tmp_path):
    srcs = [tmp_path / f"s{i}" for i in range(3)]
    dest = tmp_path / "combined" / "hotpot_questions"
    moves = plan_moves(srcs, dest)
    assert [d.name for _, d in moves] == ["sample_0001", "sample_0002", "sample_0003"]
    assert all(d.parent == dest for _, d in moves)


def test_consolidate_merges_shards_and_removes_worker_dirs(tmp_path):
    for w in range(3):
        for s in range(4):
            _make_episode(tmp_path / f"w{w}" / "uuid" / "hotpot_questions" / f"sample_{s:03d}")
    (tmp_path / "logs").mkdir()  # a non-shard sibling that must be preserved
    (tmp_path / "logs" / "sim.log").write_text("x", encoding="utf-8")
    moved, removed = consolidate(tmp_path, "run")
    assert moved == 12
    assert set(removed) == {"w0", "w1", "w2"}
    # All 12 episodes now live under the single run dir.
    combined = tmp_path / "run" / "hotpot_questions"
    assert len(list(combined.glob("sample_*"))) == 12
    assert not (tmp_path / "w0").exists()
    # Non-shard siblings (logs) are never removed.
    assert (tmp_path / "logs" / "sim.log").exists()


def test_consolidate_is_idempotent(tmp_path):
    _make_episode(tmp_path / "w0" / "uuid" / "hotpot_questions" / "sample_000")
    assert consolidate(tmp_path, "run")[0] == 1
    # Second run finds nothing left to move.
    assert consolidate(tmp_path, "run") == (0, [])
