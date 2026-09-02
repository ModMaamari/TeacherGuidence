"""
ChatNoir web search retrieval component.
"""

import time
import httpx
from typing import Dict, Any, List
from loguru import logger

from agentsim.components.base import ComponentSpec, ComponentResult, ComponentRegistry
from agentsim.components.retrieval.base import RetrievalComponent
from agentsim.workflow.context import WorkflowContext, EvidenceSpan
from agentsim.config import config


@ComponentRegistry.register("chatnoir_retriever")
class ChatNoirRetriever(RetrievalComponent):
    """
    Retrieves documents using ChatNoir web search API.
    
    This component provides access to large web corpora
    through the ChatNoir search engine.
    """
    
    def __init__(self, config: Dict[str, Any] = None, chatnoir_client=None):
        """
        Initialize ChatNoir retriever.
        
        Args:
            config: Component configuration
            chatnoir_client: ChatNoir client instance
        """
        super().__init__(config)
        self.chatnoir_client = chatnoir_client
    
    @property
    def spec(self) -> ComponentSpec:
        return ComponentSpec(
            name="chatnoir_retriever",
            category=self.category,
            description="Web search retrieval using ChatNoir",
            input_keys=["query", "subgoals"],
            output_keys=["evidence_store"],
            config_schema={
                "k": {
                    "type": "integer",
                    "default": 30,
                    "min": 1,
                    "max": 100,
                    "description": "Number of results to retrieve"
                },
                "corpus": {
                    "type": "string",
                    "default": "msmarco-v2.1-segmented",
                    "description": "ChatNoir corpus to search"
                },
                "enabled": {
                    "type": "boolean",
                    "default": True,
                    "description": "Enable/disable ChatNoir retrieval"
                }
            },
            requires_llm=False
        )
    
    async def execute(self, context: WorkflowContext) -> ComponentResult:
        """
        Execute ChatNoir retrieval.
        
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
        
        # ChatNoir client is optional - we can make direct API calls
        # if not self.chatnoir_client:
        #     logger.warning("ChatNoir client not provided, using direct API calls")

        
        # Determine queries
        subgoals = context.subgoals
        queries = subgoals if len(subgoals) > 1 else [context.query]
        
        k = self.config.get("k", 30)
        corpus = self.config.get("corpus", "cw12")
        
        logger.info(f"ChatNoir: Searching {len(queries)} queries, k={k}, corpus={corpus}")
        
        retrieved_count = 0
        
        try:
            for query_idx, query in enumerate(queries):
                logger.info(f"ChatNoir query {query_idx+1}/{len(queries)}: '{query}'")
                
                # Search ChatNoir
                if self.chatnoir_client:
                    logger.debug("Using ChatNoir client")
                    results = await self.chatnoir_client.search_index(
                        query,
                        corpus_name=corpus,
                        top_k=k
                    )
                else:
                    # Direct API call to ChatNoir
                    results = await self._search_direct(query, corpus, k)
                
                # Add results to evidence store
                for result in results:
                    snippet = result.get("snippet", "")
                    if not snippet:
                        continue
                    
                    span = EvidenceSpan(
                        source="chatnoir",
                        id=result["doc_id"],
                        start=0,
                        end=len(snippet),
                        text=snippet,
                        doc_meta={
                            "title": result.get("title", ""),
                            "score": result.get("score", 0.0),
                            "query_angle": query_idx
                        }
                    )
                    context.add_evidence(span)
                    retrieved_count += 1
                
                logger.debug(f"ChatNoir query {query_idx+1}/{len(queries)}: "
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
                    "corpus": corpus
                },
                execution_time_ms=execution_time
            )
            
        except Exception as e:
            logger.error(f"ChatNoir retrieval error: {e}")
            return ComponentResult(
                success=False,
                error=str(e),
                execution_time_ms=(time.time() - start_time) * 1000
            )
    
    async def search(self, query: str, k: int) -> List[dict]:
        """Direct search method for compatibility."""
        if not self.chatnoir_client:
            return []
        
        corpus = self.config.get("corpus", "msmarco-v2.1-segmented")
        results = await self.chatnoir_client.search_index(
            query,
            corpus_name=corpus,
            top_k=k
        )
        
        return [
            {
                "id": r["doc_id"],
                "text": r.get("snippet", ""),
                "title": r.get("title", ""),
                "score": r.get("score", 0.0)
            }
            for r in results if r.get("snippet")
        ]
    
    async def _search_direct(
        self,
        query: str,
        corpus: str,
        k: int
    ) -> List[Dict[str, Any]]:
        """
        Direct API call to ChatNoir.
        """
        try:
            headers = {}
            params = {
                "query": query,
                "index": corpus,
                "size": k
            }
            
            if config.CHATNOIR_API_KEY:
                # ChatNoir API accepts the key as query parameter; include header for safety
                params["key"] = config.CHATNOIR_API_KEY
                headers["Authorization"] = f"Bearer {config.CHATNOIR_API_KEY}"
            
            async with httpx.AsyncClient(timeout=60.0) as client:
                base_url = config.CHATNOIR_BASE_URL
                url = f"{base_url}/_search"
                
                response = await client.get(url, params=params, headers=headers)
                response.raise_for_status()
                data = response.json()
                
                # Parse results
                results = []
                for hit in data.get("results", []):
                    results.append({
                        "doc_id": hit.get("trec_id", hit.get("uuid", "")),
                        "snippet": hit.get("snippet", ""),
                        "title": hit.get("title", ""),
                        "score": hit.get("score", 0.0)
                    })
                
                logger.info(f"ChatNoir retrieved {len(results)} results for '{query}' from {corpus}")
                return results
                
        except httpx.HTTPError as e:
            logger.error(f"ChatNoir API error: {e}")
            raise
        except Exception as e:
            logger.error(f"ChatNoir search failed: {e}")
            raise

