"""
AgentSim - Modular Agentic Simulation Framework

Lightweight framework for building agentic workflows that generate 
training traces for retrieval-augmented generation systems.
"""

__version__ = "1.0.0"

from agentsim.workflow.executor import WorkflowExecutor
from agentsim.workflow.loader import WorkflowLoader
from agentsim.workflow.context import WorkflowContext

__all__ = [
    "WorkflowExecutor",
    "WorkflowLoader",
    "WorkflowContext",
]

