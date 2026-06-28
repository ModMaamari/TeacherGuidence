"""
Teacher Guidance pipeline for AgentSim.

Generates step-level retrieval-agent trajectories where a student model solves a QA
task with tools while a teacher model privately evaluates each step and exposes only
a controlled level of guidance.

The modules in this package are intentionally free of heavy dependencies
(``torch`` / ``sentence-transformers`` are only pulled by the injected LLM client),
so they can be imported and unit-tested anywhere.
"""

__all__ = ["schemas"]
