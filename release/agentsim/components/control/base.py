"""
Base class for control flow components.

Control components manage workflow execution and branching.
"""

from agentsim.components.base import BaseComponent, ComponentCategory


class ControlComponent(BaseComponent):
    """
    Abstract base class for control flow components.
    
    Control components:
    - Manage workflow execution
    - Implement conditional branching
    - Handle adaptive decision-making
    - Control iteration and loops
    """
    
    @property
    def category(self) -> ComponentCategory:
        """All control components have CONTROL category."""
        return ComponentCategory.CONTROL

