"""Tests for the trajectory explorer HTTP server (API routes)."""

import json
import threading
import urllib.request

import pytest

from agentsim.teacher_guidance.viewer import data_access as da
from agentsim.teacher_guidance.viewer.server import make_server


def _write_run(tmp_path):
    ep = {
        "qid": "q1",
        "query": "where?",
        "gold_answer": "Delhi",
        "final_answer": "Delhi",
        "guidance_level": 3,
        "student_model": "s",
        "teacher_model": "t",
        "plan_review": {"enabled": True},
        "steps": [{"t": 1, "student_action": {"action": {"tool": "search"}}}],
        "final_metrics": {"exact_match": True, "f1": 1.0, "supporting_doc_recall": 1.0},
        "stop_reason": "teacher_accept",
    }
    sd = tmp_path / "sim" / "run1" / "ds" / "sample_001"
    sd.mkdir(parents=True)
    (sd / da.EPISODE_FILENAME).write_text(json.dumps(ep) + "\n", encoding="utf-8")


@pytest.fixture
def base_url(tmp_path):
    _write_run(tmp_path)
    httpd = make_server(str(tmp_path), host="127.0.0.1", port=0)
    port = httpd.server_address[1]
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{port}"
    httpd.shutdown()
    httpd.server_close()


def _get(url):
    with urllib.request.urlopen(url, timeout=5) as resp:
        return resp.status, json.loads(resp.read().decode("utf-8"))


def test_api_runs(base_url):
    status, data = _get(f"{base_url}/api/runs")
    assert status == 200
    assert len(data) == 1
    assert data[0]["run_id"] == "sim/run1"
    assert data[0]["mean_exact_match"] == 1.0


def test_api_episodes(base_url):
    status, data = _get(f"{base_url}/api/episodes?run=sim/run1")
    assert status == 200
    assert data[0]["qid"] == "q1"


def test_api_episode_full(base_url):
    status, data = _get(f"{base_url}/api/episode?run=sim/run1&qid=q1")
    assert status == 200
    assert data["final_answer"] == "Delhi"
    assert len(data["steps"]) == 1


def test_api_episode_not_found(base_url):
    import urllib.error

    with pytest.raises(urllib.error.HTTPError) as exc:
        _get(f"{base_url}/api/episode?run=sim/run1&qid=missing")
    assert exc.value.code == 404


def test_unknown_route_404(base_url):
    import urllib.error

    with pytest.raises(urllib.error.HTTPError) as exc:
        _get(f"{base_url}/nope")
    assert exc.value.code == 404
