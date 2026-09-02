"""
Base class for verification components.

Verification components check answer quality and attribution.
"""

from agentsim.components.base import BaseComponent, ComponentCategory


class VerificationComponent(BaseComponent):
    """
    Abstract base class for verification components.
    
    Verification components:
    - Fact-check claims
    - Verify attribution
    - Check answer quality
    - Validate against evidence
    """
    
    @property
    def category(self) -> ComponentCategory:
        """All verification components have VERIFICATION category."""
        return ComponentCategory.VERIFICATION

