"""
Base class for all retrieval components.

Retrieval components search external systems (OpenSearch, vector databases, etc.)
to find relevant documents or passages for a query.
"""

from abc import abstractmethod
from agentsim.components.base import BaseComponent, ComponentCategory
from typing import List


class RetrievalComponent(BaseComponent):
    """
    Abstract base class for retrieval components.
    
    Retrieval components:
    - Take queries from context
    - Search external systems
    - Return ranked results
    - Add evidence to context
    """
    
    @property
    def category(self) -> ComponentCategory:
        """All retrieval components have RETRIEVAL category."""
        return ComponentCategory.RETRIEVAL
    
    @abstractmethod
    async def search(self, query: str, k: int) -> List[dict]:
        """
        Perform a search query.
        
        Args:
            query: Search query string
            k: Number of results to return
            
        Returns:
            List of search results with text and metadata
        """
        pass

