"""
AgentSim - Modular Agentic Simulation Framework

Lightweight framework for building agentic workflows that generate 
training traces for retrieval-augmented generation systems.
"""

__version__ = "1.0.0"

# Library logging: the harness registers its components at import time and logs each one
# at DEBUG. That is useful when debugging the harness and pure noise in front of every
# script's output, so the default sink starts at INFO. Set AGENTSIM_LOG_LEVEL=DEBUG (or
# any loguru level) to see everything again.
import os as _os
import sys as _sys

try:
    from loguru import logger as _logger

    _logger.remove()
    _logger.add(_sys.stderr, level=_os.getenv("AGENTSIM_LOG_LEVEL", "INFO"))
except Exception:  # pragma: no cover - logging must never break an import
    pass

from agentsim.workflow.executor import WorkflowExecutor
from agentsim.workflow.loader import WorkflowLoader
from agentsim.workflow.context import WorkflowContext

__all__ = [
    "WorkflowExecutor",
    "WorkflowLoader",
    "WorkflowContext",
]
