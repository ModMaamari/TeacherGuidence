"""
OpenSearch BM25 retrieval component.
"""

import time
from typing import Dict, Any, List
from loguru import logger

from agentsim.components.base import ComponentSpec, ComponentResult, ComponentRegistry
from agentsim.components.retrieval.base import RetrievalComponent
from agentsim.workflow.context import WorkflowContext, EvidenceSpan


@ComponentRegistry.register("opensearch_retriever")
class OpenSearchRetriever(RetrievalComponent):
    """
    Retrieves documents using OpenSearch with BM25 ranking.
    
    This component performs keyword-based search using OpenSearch's
    BM25 algorithm. It can handle single or multi-angle search
    when subgoals are available from a planning component.
    """
    
    def __init__(self, config: Dict[str, Any] = None, opensearch_client=None):
        """
        Initialize OpenSearch retriever.
        
        Args:
            config: Component configuration
            opensearch_client: OpenSearch client instance
        """
        super().__init__(config)
        self.opensearch_client = opensearch_client
    
    @property
    def spec(self) -> ComponentSpec:
        return ComponentSpec(
            name="opensearch_retriever",
            category=self.category,
            description="BM25-based retrieval using OpenSearch",
            input_keys=["query", "subgoals"],
            output_keys=["evidence_store"],
            config_schema={
                "k": {
                    "type": "integer",
                    "default": 50,
                    "min": 1,
                    "max": 100,
                    "description": "Number of results to retrieve per query"
                },
                "multi_angle": {
                    "type": "boolean",
                    "default": True,
                    "description": "Use subgoals for multi-angle search if available"
                },
                "k_per_query": {
                    "type": "integer",
                    "default": 20,
                    "description": "Results per query in multi-angle mode"
                }
            },
            requires_llm=False
        )
    
    async def execute(self, context: WorkflowContext) -> ComponentResult:
        """
        Execute OpenSearch retrieval.
        
        Args:
            context: Workflow context
            
        Returns:
            ComponentResult with retrieval statistics
        """
        start_time = time.time()
        
        if not self.opensearch_client:
            return ComponentResult(
                success=False,
                error="OpenSearch client not provided",
                execution_time_ms=0
            )
        
        # Determine queries to use
        subgoals = context.subgoals
        use_multi_angle = self.config.get("multi_angle", True) and len(subgoals) > 1
        
        if use_multi_angle:
            queries = subgoals
            k_per_query = self.config.get("k_per_query", 20)
        else:
            queries = [context.query]
            k_per_query = self.config.get("k", 50)
        
        # Perform retrieval
        retrieved_count = 0
        queries_executed = []
        
        try:
            for query_idx, query in enumerate(queries):
                queries_executed.append(query)
                
                # Search OpenSearch
                results = await self.opensearch_client.search(query, k=k_per_query)
                
                # Add results to evidence store
                for result in results:
                    if not result.text:
                        continue
                    
                    span = EvidenceSpan(
                        source="opensearch",
                        id=f"{result.doc_id}#seg:{result.segment_id}",
                        start=0,
                        end=len(result.text),
                        text=result.text,
                        doc_meta={
                            "title": result.title,
                            "score": result.score,
                            "query_angle": query_idx
                        }
                    )
                    context.add_evidence(span)
                    retrieved_count += 1
                
                logger.debug(f"OpenSearch query {query_idx+1}/{len(queries)}: "
                           f"{len(results)} results")
            
            execution_time = (time.time() - start_time) * 1000
            
            return ComponentResult(
                success=True,
                data={
                    "retrieved_count": retrieved_count,
                    "total_evidence": len(context.evidence_store),
                    "queries_executed": len(queries_executed),
                    "multi_angle": use_multi_angle
                },
                metadata={
                    "queries": queries_executed,
                    "k_per_query": k_per_query
                },
                execution_time_ms=execution_time
            )
            
        except Exception as e:
            logger.error(f"OpenSearch retrieval error: {e}")
            return ComponentResult(
                success=False,
                error=str(e),
                execution_time_ms=(time.time() - start_time) * 1000
            )
    
    async def search(self, query: str, k: int) -> List[dict]:
        """Direct search method for compatibility."""
        if not self.opensearch_client:
            return []
        
        results = await self.opensearch_client.search(query, k=k)
        return [
            {
                "id": f"{r.doc_id}#seg:{r.segment_id}",
                "text": r.text,
                "title": r.title,
                "score": r.score
            }
            for r in results if r.text
        ]

