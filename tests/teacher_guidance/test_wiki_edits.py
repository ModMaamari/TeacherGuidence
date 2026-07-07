"""Tests for the surgical wiki edit engine (agentsim/teacher_guidance/wiki.py)."""

from agentsim.teacher_guidance.wiki import (
    MAX_FACTS,
    WikiDoc,
    apply_wiki_edits,
    parse_wiki,
    render_wiki,
    render_wiki_numbered,
)

WIKI = """FACTS:
- Jim Price toured with The Rolling Stones (q1::doc0)
- Goats Head Soup was released in 1973 (q1::doc6)
ANSWER: 1973
NEXT: verify the release month"""


# ---------------------------------------------------------------------------
# parse / render
# ---------------------------------------------------------------------------
def test_parse_and_render_roundtrip():
    doc = parse_wiki(WIKI)
    assert len(doc.facts) == 2 and doc.answer == "1973"
    assert doc.next_step == "verify the release month"
    assert render_wiki(doc) == WIKI


def test_parse_tolerates_numbered_and_star_bullets():
    doc = parse_wiki("FACTS:\n1. fact one (d1)\n2) fact two (d2)\n* fact three (d3)")
    assert doc.facts == ["fact one (d1)", "fact two (d2)", "fact three (d3)"]


def test_render_numbered_for_edit_prompt():
    numbered = render_wiki_numbered(WIKI)
    assert "1. Jim Price" in numbered and "2. Goats Head Soup" in numbered
    assert render_wiki_numbered("") == "(empty)"


def test_render_skips_unset_sections():
    assert render_wiki(WikiDoc(facts=["f (d)"])) == "FACTS:\n- f (d)"
    assert render_wiki(WikiDoc()) == ""


# ---------------------------------------------------------------------------
# edit commands
# ---------------------------------------------------------------------------
def test_add_edit_del_answer_next():
    new, applied, ignored = apply_wiki_edits(
        WIKI,
        "ADD: the album was by The Rolling Stones (q1::doc6)\n"
        "EDIT 2: Goats Head Soup was released in August 1973 (q1::doc6)\n"
        "ANSWER: August 1973\n"
        "NEXT: finish",
    )
    assert not ignored
    assert "August 1973 (q1::doc6)" in new and "the album was by The Rolling Stones" in new
    assert "ANSWER: August 1973" in new and "NEXT: finish" in new
    assert [a.split()[0] for a in applied] == ["ADD", "EDIT", "ANSWER", "NEXT"]

    new2, applied2, _ = apply_wiki_edits(new, "DEL 1")
    assert "the album was by The Rolling Stones" in new2  # line 2 became line 1's successor
    assert "Jim Price" not in new2
    assert applied2 == ["DEL 1"]


def test_keep_is_a_noop():
    new, applied, ignored = apply_wiki_edits(WIKI, "KEEP")
    assert new == WIKI and applied == ["KEEP"] and not ignored


def test_duplicate_add_skipped():
    new, applied, ignored = apply_wiki_edits(
        WIKI, "ADD: Jim Price toured with The Rolling Stones (q1::doc0)"
    )
    assert new == WIKI and not applied
    assert ignored and "duplicate" in ignored[0]


def test_bad_indices_and_garbage_ignored():
    new, applied, ignored = apply_wiki_edits(WIKI, "DEL 9\nEDIT 0: x\ntotally not a command")
    assert new == WIKI and not applied and len(ignored) == 3


def test_placeholder_answer_rejected():
    for bad in ("ANSWER: ?", "ANSWER: unknown", "ANSWER: n/a"):
        new, applied, ignored = apply_wiki_edits(WIKI, bad)
        assert "ANSWER: 1973" in new and not applied and ignored


def test_facts_capped_oldest_dropped():
    wiki = "FACTS:\n" + "\n".join(f"- fact {i} (d{i})" for i in range(MAX_FACTS))
    new, applied, _ = apply_wiki_edits(wiki, "ADD: the newest fact (d9)")
    doc = parse_wiki(new)
    assert len(doc.facts) == MAX_FACTS
    assert doc.facts[-1] == "the newest fact (d9)"
    assert "fact 0" not in new  # oldest dropped
    assert any(a.startswith("trimmed") for a in applied)


def test_full_rewrite_fallback():
    # A model that ignores the protocol and emits a whole document is accepted.
    new, applied, ignored = apply_wiki_edits(
        WIKI, "FACTS:\n- brand new fact (d1)\nANSWER: something\nNEXT: done"
    )
    assert applied == ["rewrite"]
    assert "brand new fact" in new and "Jim Price" not in new


def test_edits_on_empty_wiki_build_it_up():
    new, applied, ignored = apply_wiki_edits(
        "", "ADD: first fact (d1)\nANSWER: candidate\nNEXT: search more"
    )
    assert not ignored
    assert new == "FACTS:\n- first fact (d1)\nANSWER: candidate\nNEXT: search more"


def test_code_fence_stripped_from_script():
    new, applied, _ = apply_wiki_edits(WIKI, "```\nANSWER: August 1973\n```")
    assert "ANSWER: August 1973" in new and applied == ["ANSWER"]
