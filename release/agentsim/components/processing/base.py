"""
Base class for processing components.

Processing components filter, transform, and refine evidence.
"""

from agentsim.components.base import BaseComponent, ComponentCategory


class ProcessingComponent(BaseComponent):
    """
    Abstract base class for processing components.
    
    Processing components:
    - Rerank search results
    - Remove duplicates
    - Filter by quality
    - Extract key evidence spans
    """
    
    @property
    def category(self) -> ComponentCategory:
        """All processing components have PROCESSING category."""
        return ComponentCategory.PROCESSING

