"""
Base class for synthesis components.

Synthesis components generate answers from collected evidence.
"""

from agentsim.components.base import BaseComponent, ComponentCategory


class SynthesisComponent(BaseComponent):
    """
    Abstract base class for synthesis components.
    
    Synthesis components:
    - Take evidence from context
    - Generate answers
    - Add citations
    - Format output
    """
    
    @property
    def category(self) -> ComponentCategory:
        """All synthesis components have SYNTHESIS category."""
        return ComponentCategory.SYNTHESIS

