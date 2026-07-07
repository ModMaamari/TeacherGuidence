"""Tests for the per-episode agent wiki (wiki_read/wiki_write)."""

import json

import pytest

from agentsim.workflow.context import WorkflowContext
from agentsim.teacher_guidance.local_retrieval import HotpotLocalRetriever
from agentsim.teacher_guidance.schemas import StudentAction, TOOLS, DECISION_CATEGORIES
from agentsim.teacher_guidance.tool_executor import execute_student_tool
from agentsim.teacher_guidance.prompts import (
    build_student_visible_state,
    build_student_prompt,
    build_wiki_update_prompt,
)
from agentsim.teacher_guidance.tool_executor import clean_wiki_content
from agentsim.teacher_guidance.schemas import GuidanceConfig
from agentsim.teacher_guidance.json_utils import parse_student_action
from agentsim.teacher_guidance.pydantic_schemas import (
    StudentActionGenerationModel,
    StudentActionWikiGenerationModel,
)


CORPUS = [
    {"doc_id": "q1::doc0", "qid": "q1", "title": "Doc",
     "text": "Some text.", "sentences": ["Some text."],
     "is_gold_doc": True, "gold_sent_ids": [0], "source": "hotpotqa", "split": "validation"},
]


@pytest.fixture
def retriever(tmp_path):
    path = tmp_path / "corpus.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        for row in CORPUS:
            f.write(json.dumps(row) + "\n")
    return HotpotLocalRetriever(str(path))


@pytest.fixture
def context():
    return WorkflowContext(
        task_id="q1",
        query="A question?",
        metadata={"retrieval_scope": {"qid": "q1"}, "wiki_enabled": True},
    )


def _action(tool, params):
    return StudentAction.from_dict({"action": {"tool": tool, "params": params}})


# ---------------------------------------------------------------------------
# Executor
# ---------------------------------------------------------------------------
def test_wiki_vocab_registered():
    assert "wiki_read" in TOOLS and "wiki_write" in TOOLS
    assert "manage_wiki" in DECISION_CATEGORIES


def test_wiki_write_then_read(context, retriever):
    obs = execute_student_tool(context, _action("wiki_write", {"content": "key fact: X"}), retriever)
    assert obs["status"] == "ok" and obs["chars"] == len("key fact: X")
    assert context.metadata["wiki"] == "key fact: X"

    obs = execute_student_tool(context, _action("wiki_read", {}), retriever)
    assert obs["status"] == "ok" and obs["content"] == "key fact: X"
    assert context.metadata["wiki_just_read"] == "key fact: X"


def test_wiki_write_replaces_whole_file(context, retriever):
    execute_student_tool(context, _action("wiki_write", {"content": "old"}), retriever)
    execute_student_tool(context, _action("wiki_write", {"content": "new"}), retriever)
    assert context.metadata["wiki"] == "new"


def test_wiki_read_on_empty_wiki(context, retriever):
    obs = execute_student_tool(context, _action("wiki_read", {}), retriever)
    assert obs["status"] == "ok" and obs["content"] == "" and obs["chars"] == 0


def test_wiki_just_read_is_one_shot(context, retriever):
    execute_student_tool(context, _action("wiki_write", {"content": "note"}), retriever)
    execute_student_tool(context, _action("wiki_read", {}), retriever)
    assert context.metadata.get("wiki_just_read") == "note"
    # The next action retires the pending read.
    execute_student_tool(context, _action("synthesize", {}), retriever)
    assert "wiki_just_read" not in context.metadata


# ---------------------------------------------------------------------------
# Student-visible state + prompt
# ---------------------------------------------------------------------------
def test_state_includes_wiki_fields_only_when_enabled():
    ctx = WorkflowContext(task_id="q1", query="q", metadata={"wiki": "notes"})
    state = build_student_visible_state(ctx, 1, 5)
    assert "wiki_enabled" not in state and "wiki_chars" not in state

    ctx.metadata["wiki_enabled"] = True
    state = build_student_visible_state(ctx, 1, 5)
    assert state["wiki_enabled"] is True and state["wiki_chars"] == len("notes")


def test_state_surfaces_wiki_content_after_read():
    ctx = WorkflowContext(
        task_id="q1", query="q",
        metadata={"wiki_enabled": True, "wiki": "notes", "wiki_just_read": "notes"},
    )
    state = build_student_visible_state(ctx, 2, 5)
    assert state["wiki_content"] == "notes"


def test_prompt_mentions_wiki_when_enabled():
    state = {"question": "q", "step": 1, "budget": 5, "wiki_enabled": True, "wiki_chars": 0}
    p = build_student_prompt(state, GuidanceConfig(level=3), force_finish=False)
    assert "wiki_read" in p and "wiki_write" in p and "wiki.md" in p
    assert "currently empty" in p


def test_prompt_shows_wiki_size_and_content(retriever):
    state = {"question": "q", "step": 2, "budget": 5, "wiki_enabled": True, "wiki_chars": 5}
    p = build_student_prompt(state, GuidanceConfig(level=3), force_finish=False)
    assert "5 characters of saved notes" in p

    state["wiki_content"] = "my saved note"
    p = build_student_prompt(state, GuidanceConfig(level=3), force_finish=False)
    assert "my saved note" in p


def test_prompt_omits_wiki_when_disabled():
    state = {"question": "q", "step": 1, "budget": 5}
    p = build_student_prompt(state, GuidanceConfig(level=3), force_finish=False)
    assert "wiki" not in p.lower()


# ---------------------------------------------------------------------------
# Auto wiki mode (wiki_mode == "auto")
# ---------------------------------------------------------------------------
def test_auto_mode_state_includes_wiki_content_every_step():
    ctx = WorkflowContext(
        task_id="q1", query="q",
        metadata={"wiki_enabled": True, "wiki_mode": "auto", "wiki": "notes"},
    )
    state = build_student_visible_state(ctx, 3, 5)
    assert state["wiki_mode"] == "auto" and state["wiki_content"] == "notes"
    assert "wiki_chars" not in state


def test_auto_mode_prompt_shows_wiki_but_no_wiki_tools():
    state = {"question": "q", "step": 1, "budget": 5,
             "wiki_enabled": True, "wiki_mode": "auto", "wiki_content": "saved note"}
    p = build_student_prompt(state, GuidanceConfig(level=3), force_finish=False)
    assert "saved note" in p and "wiki.md" in p
    assert "wiki_read" not in p and "wiki_write" not in p


def test_wiki_update_prompt_contents():
    state = {"question": "who?", "wiki_content": "old note"}
    p = build_wiki_update_prompt(
        state,
        {"action": {"tool": "search", "params": {"query": "x"}}},
        {"tool": "search", "status": "ok"},
    )
    assert "who?" in p and "old note" in p and "search" in p
    assert "NEW full content" in p


def test_clean_wiki_content_strips_fences_and_caps_length():
    assert clean_wiki_content("```markdown\nnote\n```") == "note"
    assert clean_wiki_content("  plain  ") == "plain"
    assert len(clean_wiki_content("x" * 5000)) == 2000
    assert clean_wiki_content(None) == ""


def test_auto_mode_component_updates_wiki_after_step(tmp_path):
    import asyncio
    from agentsim.components.control.teacher_guided_agent_step import TeacherGuidedAgentStep

    student_json = json.dumps({
        "thought": "search for the relevant document first",
        "decision": {"category": "need_retrieval", "parametric_knowledge_used": False},
        "action": {"tool": "search", "params": {"query": "Doc", "k": 1}},
        "new_facts_extracted": [],
    })
    teacher_json = json.dumps({
        "guidance_level": 3,
        "student_visible": {"score_binary": 1, "score_continuous": 0.9, "feedback": "good", "hint": None},
        "private_diagnosis": {},
        "teacher_decision": "continue",
    })

    class StubLLM:
        async def get_completion(self, prompt, model=None, temperature=0.0, max_tokens=None, **kw):
            if "update the wiki" in prompt.lower() or "NEW full content" in prompt:
                return "key fact: Doc found"
            if "teacher evaluating" in prompt:
                return teacher_json
            return student_json

    corpus_path = tmp_path / "corpus.jsonl"
    with open(corpus_path, "w", encoding="utf-8") as f:
        for row in CORPUS:
            f.write(json.dumps(row) + "\n")
    ctx = WorkflowContext(
        task_id="q1", query="A question?",
        metadata={
            "corpus_path": str(corpus_path),
            "student_model": "stub", "teacher_model": "stub",
            "guidance": {"level": 3},
            "gold": {"answer": "x", "gold_doc_ids": ["q1::doc0"]},
            "retrieval_scope": {"qid": "q1"},
            "wiki_enabled": True, "wiki_mode": "auto",
        },
    )
    comp = TeacherGuidedAgentStep({"step_index": 1, "budget": 5}, llm_client=StubLLM())
    result = asyncio.run(comp.execute(ctx))
    assert result.success
    assert ctx.metadata["wiki"] == "key fact: Doc found"
    step = ctx.metadata["teacher_guided_steps"][0]
    assert step["wiki_after"] == "key fact: Doc found"
    assert step["wiki_update_call"] is not None


# ---------------------------------------------------------------------------
# Schemas / validation
# ---------------------------------------------------------------------------
def test_wiki_action_parses_and_validates():
    raw = json.dumps({
        "thought": "Saving the key fact I just confirmed.",
        "decision": {"category": "manage_wiki", "parametric_knowledge_used": False},
        "action": {"tool": "wiki_write", "params": {"content": "fact"}},
        "new_facts_extracted": [],
    })
    action, info = parse_student_action(raw)
    assert info["json_valid"] and info["action_valid"]
    assert action.action.tool == "wiki_write"


def test_generation_schema_includes_wiki_tools_only_in_wiki_variant():
    base = json.dumps(StudentActionGenerationModel.model_json_schema())
    wiki = json.dumps(StudentActionWikiGenerationModel.model_json_schema())
    assert "wiki_write" not in base and "wiki_read" not in base
    assert "wiki_write" in wiki and "wiki_read" in wiki


def test_wiki_generation_schema_requires_content():
    with pytest.raises(Exception):
        StudentActionWikiGenerationModel.model_validate({
            "thought": "writing an empty note should fail",
            "action": {"tool": "wiki_write", "params": {"content": ""}},
        })
    ok = StudentActionWikiGenerationModel.model_validate({
        "thought": "reading my notes back before answering",
        "action": {"tool": "wiki_read", "params": {}},
    })
    assert ok.action.tool == "wiki_read"
