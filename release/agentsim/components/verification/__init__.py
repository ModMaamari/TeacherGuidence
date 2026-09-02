"""
Verification components for fact-checking and validation.

This category includes components that verify answer quality,
check attribution, and validate claims.
"""

from agentsim.components.verification.base import VerificationComponent
from agentsim.components.verification.fact_checker import FactCheckerComponent
from agentsim.components.verification.attribution import AttributionGateComponent

__all__ = [
    "VerificationComponent",
    "FactCheckerComponent",
    "AttributionGateComponent",
]

