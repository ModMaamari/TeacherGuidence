"""
Component library for building agentic workflows.

Components are organized by category:
- retrieval: Search and retrieval components
- processing: Data processing and filtering components
- planning: Query decomposition and planning components
- verification: Fact-checking and verification components
- synthesis: Answer generation and synthesis components
- control: Control flow and conditional components
"""

from agentsim.components.base import BaseComponent, ComponentCategory

# Import all components to register them
from agentsim.components.retrieval import opensearch, vector, chatnoir
from agentsim.components.processing import reranker, deduplicator, filter, extractor
from agentsim.components.planning import planner, query_formulator
from agentsim.components.synthesis import answer_drafter, finalizer
from agentsim.components.verification import fact_checker, attribution
from agentsim.components.control import (
    condition,
    teacher_guided_agent_step,
    teacher_guided_plan_review,
)

__all__ = ["BaseComponent", "ComponentCategory"]

