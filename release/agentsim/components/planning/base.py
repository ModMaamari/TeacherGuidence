"""
Base class for planning components.

Planning components decompose queries and create search strategies.
"""

from agentsim.components.base import BaseComponent, ComponentCategory


class PlanningComponent(BaseComponent):
    """
    Abstract base class for planning components.
    
    Planning components:
    - Analyze the query
    - Decompose into sub-tasks
    - Generate search strategies
    - Create execution plans
    """
    
    @property
    def category(self) -> ComponentCategory:
        """All planning components have PLANNING category."""
        return ComponentCategory.PLANNING

