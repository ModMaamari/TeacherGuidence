"""
Synthesis components for generating answers from evidence.

This category includes components that synthesize final answers
from collected evidence.
"""

from agentsim.components.synthesis.base import SynthesisComponent
from agentsim.components.synthesis.answer_drafter import AnswerDrafterComponent
from agentsim.components.synthesis.finalizer import FinalizerComponent

__all__ = [
    "SynthesisComponent",
    "AnswerDrafterComponent",
    "FinalizerComponent",
]

