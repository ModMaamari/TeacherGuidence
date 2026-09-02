"""
Vector/semantic search retrieval component.
"""

import time
from typing import Dict, Any, List
from loguru import logger

from agentsim.components.base import ComponentSpec, ComponentResult, ComponentRegistry
from agentsim.components.retrieval.base import RetrievalComponent
from agentsim.workflow.context import WorkflowContext, EvidenceSpan


@ComponentRegistry.register("vector_retriever")
class VectorRetriever(RetrievalComponent):
    """
    Retrieves documents using vector/semantic search with embeddings.
    
    This component performs semantic similarity search using
    dense embeddings and k-NN search.
    """
    
    def __init__(self, config: Dict[str, Any] = None, vector_client=None, llm_client=None):
        """
        Initialize vector retriever.
        
        Args:
            config: Component configuration
            vector_client: Vector search client instance
            llm_client: LLM client for generating embeddings
        """
        super().__init__(config)
        self.vector_client = vector_client
        self.llm_client = llm_client
    
    @property
    def spec(self) -> ComponentSpec:
        return ComponentSpec(
            name="vector_retriever",
            category=self.category,
            description="Semantic retrieval using vector embeddings",
            input_keys=["query", "subgoals"],
            output_keys=["evidence_store"],
            config_schema={
                "k": {
                    "type": "integer",
                    "default": 25,
                    "min": 1,
                    "max": 100,
                    "description": "Number of results to retrieve"
                },
                "num_candidates": {
                    "type": "integer",
                    "default": 100,
                    "description": "Number of candidates for k-NN search"
                },
                "enabled": {
                    "type": "boolean",
                    "default": True,
                    "description": "Enable/disable vector retrieval"
                }
            },
            requires_llm=True
        )
    
    async def execute(self, context: WorkflowContext) -> ComponentResult:
        """
        Execute vector retrieval.
        
        Args:
            context: Workflow context
            
        Returns:
            ComponentResult with retrieval statistics
        """
        start_time = time.time()
        
        if not self.config.get("enabled", True):
            return ComponentResult(
                success=True,
                data={"skipped": True},
                metadata={"reason": "disabled"},
                execution_time_ms=(time.time() - start_time) * 1000
            )
        
        if not self.vector_client or not self.llm_client:
            return ComponentResult(
                success=False,
                error="Vector client or LLM client not provided",
                execution_time_ms=0
            )
        
        # Use recent queries or main query
        queries = context.queries[-2:] if context.queries else [context.query]
        queries = [q for q in queries if q]  # Filter empty
        
        if not queries:
            queries = [context.query]
        
        k = self.config.get("k", 25)
        num_candidates = self.config.get("num_candidates", 100)
        
        retrieved_count = 0
        
        try:
            for query_idx, query in enumerate(queries):
                # Get embedding
                embedding = await self.llm_client.get_embedding(query)
                
                # Perform k-NN search
                results = await self.vector_client.knn_search(
                    embedding,
                    k=k,
                    num_candidates=num_candidates
                )
                
                # Add results to evidence store
                for hit in results:
                    span = EvidenceSpan(
                        source="vector",
                        id=hit.doc_id,
                        start=0,
                        end=len(hit.text),
                        text=hit.text,
                        doc_meta={
                            "score": hit.score,
                            "query_angle": query_idx
                        }
                    )
                    context.add_evidence(span)
                    retrieved_count += 1
                
                logger.debug(f"Vector query {query_idx+1}/{len(queries)}: "
                           f"{len(results)} results")
            
            execution_time = (time.time() - start_time) * 1000
            
            return ComponentResult(
                success=True,
                data={
                    "retrieved_count": retrieved_count,
                    "total_evidence": len(context.evidence_store),
                    "queries_executed": len(queries)
                },
                metadata={
                    "queries": queries,
                    "k": k,
                    "num_candidates": num_candidates
                },
                execution_time_ms=execution_time
            )
            
        except Exception as e:
            logger.error(f"Vector retrieval error: {e}")
            return ComponentResult(
                success=False,
                error=str(e),
                execution_time_ms=(time.time() - start_time) * 1000
            )
    
    async def search(self, query: str, k: int) -> List[dict]:
        """Direct search method for compatibility."""
        if not self.vector_client or not self.llm_client:
            return []
        
        embedding = await self.llm_client.get_embedding(query)
        results = await self.vector_client.knn_search(embedding, k=k)
        
        return [
            {
                "id": r.doc_id,
                "text": r.text,
                "score": r.score
            }
            for r in results
        ]

