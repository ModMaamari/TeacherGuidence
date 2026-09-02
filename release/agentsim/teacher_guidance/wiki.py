"""
Structured wiki.md model + surgical edit engine for the agent wiki (auto mode).

The wiki is a small templated document (FACTS / ANSWER / NEXT). Instead of asking the
student to regenerate the whole file after every step (slow, token-hungry, and lossy --
a rewrite can silently drop good facts), the post-step call emits short edit commands
that this module applies deterministically:

    ADD: <new fact> (doc_id)      append a FACTS line (duplicates skipped)
    EDIT <n>: <corrected fact>    replace numbered FACTS line n
    DEL <n>                       delete numbered FACTS line n
    ANSWER: <best guess>          set the running best answer ("?"/"unknown" rejected)
    NEXT: <next step>             set the next-step note
    KEEP                          explicit no-op

Robustness rules: unknown/malformed lines are ignored (and reported), indices out of
range are ignored, FACTS is capped (oldest dropped first), and if the model ignores the
protocol and emits a full FACTS/ANSWER/NEXT document anyway, that output is accepted as
a whole-file rewrite -- graceful degradation, an episode never breaks on a bad edit.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Tuple

from agentsim.teacher_guidance.tool_executor import clean_wiki_content

MAX_FACTS = 6
_PLACEHOLDER_ANSWERS = {"", "?", "unknown", "n/a", "none"}


@dataclass
class WikiDoc:
    facts: List[str] = field(default_factory=list)
    answer: str = ""
    next_step: str = ""


def parse_wiki(text: str) -> WikiDoc:
    """Parse wiki.md text into its FACTS/ANSWER/NEXT parts (tolerant of formatting)."""
    doc = WikiDoc()
    in_facts = False
    for raw in str(text or "").splitlines():
        line = raw.strip()
        if not line:
            continue
        low = line.lower()
        if low.startswith("facts:"):
            in_facts = True
            rest = line.split(":", 1)[1].strip()
            if rest:
                doc.facts.append(rest)
        elif low.startswith("answer:"):
            in_facts = False
            doc.answer = line.split(":", 1)[1].strip()
        elif low.startswith("next:"):
            in_facts = False
            doc.next_step = line.split(":", 1)[1].strip()
        elif in_facts:
            # "- fact", "1. fact", "2) fact", or a bare continuation line.
            doc.facts.append(re.sub(r"^(-|\*|\d+[.)])\s*", "", line))
    return doc


def render_wiki(doc: WikiDoc) -> str:
    """Canonical wiki.md text ("- " fact bullets; ANSWER/NEXT only when set)."""
    parts: List[str] = []
    if doc.facts:
        parts.append("FACTS:")
        parts.extend(f"- {f}" for f in doc.facts[-MAX_FACTS:])
    if doc.answer:
        parts.append(f"ANSWER: {doc.answer}")
    if doc.next_step:
        parts.append(f"NEXT: {doc.next_step}")
    return "\n".join(parts)


def render_wiki_numbered(text: str) -> str:
    """The wiki as shown in the edit prompt: FACTS lines numbered for EDIT/DEL."""
    doc = parse_wiki(text)
    if not (doc.facts or doc.answer or doc.next_step):
        # Non-template content (e.g. a legacy free-form wiki): show it raw rather
        # than pretending the wiki is empty.
        return str(text or "").strip() or "(empty)"
    parts: List[str] = []
    if doc.facts:
        parts.append("FACTS:")
        parts.extend(f"{i}. {f}" for i, f in enumerate(doc.facts, 1))
    if doc.answer:
        parts.append(f"ANSWER: {doc.answer}")
    if doc.next_step:
        parts.append(f"NEXT: {doc.next_step}")
    return "\n".join(parts)


def _substantive_answer(value: str) -> str:
    v = value.strip().strip('"').strip()
    return "" if v.lower() in _PLACEHOLDER_ANSWERS else v


_CMD_RE = re.compile(
    r"^(?:[-*]\s*)?(?:"
    r"(?P<add>add)\s*:?\s*(?P<add_text>.+)"
    r"|(?P<edit>edit)\s*#?(?P<edit_n>\d+)\s*:?\s*(?P<edit_text>.+)"
    r"|(?P<del>del(?:ete)?)\s*#?(?P<del_n>\d+)\s*$"
    r"|(?P<answer>answer)\s*:\s*(?P<answer_text>.+)"
    r"|(?P<next>next)\s*:\s*(?P<next_text>.+)"
    r"|(?P<keep>keep)\s*$"
    r")",
    re.IGNORECASE,
)


def apply_wiki_edits(wiki_text: str, edit_script: str) -> Tuple[str, List[str], List[str]]:
    """Apply an edit script to the wiki; returns (new_text, applied, ignored).

    ``applied`` holds one normalized entry per executed command; ``ignored`` holds the
    lines that were skipped and why -- both are logged/exported for analysis. If the
    script contains no valid command but looks like a full wiki document, it is treated
    as a whole-file rewrite (``rewrite`` appears in ``applied``)."""
    script = clean_wiki_content(edit_script)
    # A script whose first non-empty line is "FACTS:" is a full document, not a command
    # list (a document's ANSWER:/NEXT: lines would otherwise half-apply as commands) --
    # accept it as a whole-file rewrite rather than losing the update.
    first_line = next((ln.strip().lower() for ln in script.splitlines() if ln.strip()), "")
    if first_line.startswith("facts:"):
        return clean_wiki_content(render_wiki(parse_wiki(script))), ["rewrite"], []

    doc = parse_wiki(wiki_text)
    applied: List[str] = []
    ignored: List[str] = []

    for raw in script.splitlines():
        line = raw.strip()
        if not line:
            continue
        m = _CMD_RE.match(line)
        if not m:
            ignored.append(f"unrecognized: {line[:80]}")
            continue
        if m.group("add"):
            fact = m.group("add_text").strip()
            norm = re.sub(r"\s+", " ", fact.lower())
            if any(re.sub(r"\s+", " ", f.lower()) == norm for f in doc.facts):
                ignored.append(f"duplicate: {fact[:80]}")
            else:
                doc.facts.append(fact)
                applied.append(f"ADD {fact[:80]}")
        elif m.group("edit"):
            n = int(m.group("edit_n"))
            if 1 <= n <= len(doc.facts):
                doc.facts[n - 1] = m.group("edit_text").strip()
                applied.append(f"EDIT {n}")
            else:
                ignored.append(f"bad index: {line[:80]}")
        elif m.group("del"):
            n = int(m.group("del_n"))
            if 1 <= n <= len(doc.facts):
                doc.facts.pop(n - 1)
                applied.append(f"DEL {n}")
            else:
                ignored.append(f"bad index: {line[:80]}")
        elif m.group("answer"):
            value = _substantive_answer(m.group("answer_text"))
            if value:
                doc.answer = value
                applied.append("ANSWER")
            else:
                ignored.append("placeholder answer rejected")
        elif m.group("next"):
            doc.next_step = m.group("next_text").strip()
            applied.append("NEXT")
        elif m.group("keep"):
            applied.append("KEEP")

    if len(doc.facts) > MAX_FACTS:  # cap: drop oldest facts first
        dropped = len(doc.facts) - MAX_FACTS
        doc.facts = doc.facts[dropped:]
        applied.append(f"trimmed {dropped} oldest fact(s)")

    return clean_wiki_content(render_wiki(doc)), applied, ignored
