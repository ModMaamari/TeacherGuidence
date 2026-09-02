"""
Retrieval components for searching and retrieving evidence.

This category includes all components that interact with
search systems to retrieve relevant documents and passages.
"""

from agentsim.components.retrieval.base import RetrievalComponent
from agentsim.components.retrieval.opensearch import OpenSearchRetriever
from agentsim.components.retrieval.vector import VectorRetriever
from agentsim.components.retrieval.chatnoir import ChatNoirRetriever

__all__ = [
    "RetrievalComponent",
    "OpenSearchRetriever",
    "VectorRetriever",
    "ChatNoirRetriever",
]

