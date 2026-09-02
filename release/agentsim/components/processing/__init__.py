"""
Processing components for filtering and transforming evidence.

This category includes components that process retrieved evidence
through reranking, deduplication, filtering, and extraction.
"""

from agentsim.components.processing.base import ProcessingComponent
from agentsim.components.processing.reranker import RerankerComponent
from agentsim.components.processing.deduplicator import DeduplicatorComponent
from agentsim.components.processing.filter import FilterComponent
from agentsim.components.processing.extractor import ExtractorComponent

__all__ = [
    "ProcessingComponent",
    "RerankerComponent",
    "DeduplicatorComponent",
    "FilterComponent",
    "ExtractorComponent",
]

