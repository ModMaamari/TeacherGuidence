"""
Reranker component for fusing and reranking results.
"""

import time
from typing import Dict, Any, List
from loguru import logger

from agentsim.components.base import ComponentSpec, ComponentResult, ComponentRegistry
from agentsim.components.processing.base import ProcessingComponent
from agentsim.workflow.context import WorkflowContext


@ComponentRegistry.register("reranker")
class RerankerComponent(ProcessingComponent):
    """
    Reranks and fuses results from multiple retrieval sources.
    
    Supports multiple reranking methods:
    - cross_encoder: Uses sentence-transformers cross-encoder models
    - bm25: Uses BM25 algorithm for keyword-based reranking
    - simple: Simple score-based fusion (fast, lightweight)
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self._reranker_model = None
        self._bm25 = None
    
    @property
    def spec(self) -> ComponentSpec:
        return ComponentSpec(
            name="reranker",
            category=self.category,
            description="Reranks and fuses multi-source results",
            input_keys=["evidence_store"],
            output_keys=["evidence_store"],
            config_schema={
                "method": {
                    "type": "string",
                    "default": "cross_encoder",
                    "enum": ["cross_encoder", "bm25", "simple"],
                    "description": "Reranking method"
                },
                "model": {
                    "type": "string",
                    "default": "cross-encoder/ms-marco-MiniLM-L-6-v2",
                    "description": "Model for cross-encoder method"
                },
                "top_k": {
                    "type": "integer",
                    "default": 20,
                    "min": 1,
                    "max": 100,
                    "description": "Number of top results to keep"
                }
            },
            requires_llm=False
        )
    
    async def execute(self, context: WorkflowContext) -> ComponentResult:
        """
        Execute reranking.
        
        Args:
            context: Workflow context
            
        Returns:
            ComponentResult with reranking statistics
        """
        start_time = time.time()
        
        method = self.config.get("method", "cross_encoder")
        top_k = self.config.get("top_k", 20)
        original_count = len(context.evidence_store)
        
        if original_count == 0:
            return ComponentResult(
                success=True,
                data={"original_count": 0, "reranked_count": 0, "method": method},
                execution_time_ms=(time.time() - start_time) * 1000
            )
        
        query = context.query
        
        # Rerank based on method
        if method == "cross_encoder":
            scored_spans = await self._rerank_cross_encoder(query, context.evidence_store)
        elif method == "bm25":
            scored_spans = self._rerank_bm25(query, context.evidence_store)
        else:  # simple
            scored_spans = self._rerank_simple(context.evidence_store)
        
        # Sort and take top-k
        scored_spans.sort(reverse=True, key=lambda x: x[0])
        reranked_spans = scored_spans[:top_k]
        context.evidence_store = [span for score, span in reranked_spans]
        
        execution_time = (time.time() - start_time) * 1000
        
        logger.info(f"Reranked {original_count} -> {len(context.evidence_store)} spans using {method}")
        
        # Generate detailed private reasoning about the reranking process
        if method == "cross_encoder":
            model_name = self.config.get("model", "cross-encoder/ms-marco-MiniLM-L-6-v2")
            top_scores = [f"{score:.4f}" for score, _ in reranked_spans[:3]]
            
            # Get snippets of top reranked documents
            top_snippets = [span.text[:80] + "..." for _, span in reranked_spans[:3]]
            
            private_reasoning = (
                f"Applied {model_name} cross-encoder to rerank {original_count} retrieved evidence spans. "
                f"Cross-encoder jointly encodes query and each document to compute semantic relevance scores. "
                f"Top 3 scores: {', '.join(top_scores)}. "
                f"Highest-ranked evidence: '{top_snippets[0]}' (score: {top_scores[0]}). "
                f"The model identified these as most semantically relevant to the query '{query}'. "
                f"Filtered from {original_count} to top {len(context.evidence_store)} spans for downstream synthesis."
            )
        elif method == "bm25":
            top_snippets = [span.text[:80] + "..." for _, span in reranked_spans[:3]]
            private_reasoning = (
                f"Applied BM25 lexical matching to rerank {original_count} evidence spans. "
                f"BM25 scores based on term frequency-inverse document frequency with document length normalization. "
                f"Top-ranked passage: '{top_snippets[0]}'. "
                f"Selected {len(context.evidence_store)} spans with highest keyword overlap with query '{query}'."
            )
        else:
            private_reasoning = (
                f"Applied simple score-based reranking to {original_count} evidence spans. "
                f"Combined retrieval scores (70%) with position-based recency (30%). "
                f"Selected top {len(context.evidence_store)} spans for synthesis."
            )
        
        return ComponentResult(
            success=True,
            data={
                "original_count": original_count,
                "reranked_count": len(context.evidence_store),
                "top_k": top_k,
                "method": method
            },
            metadata={
                "method": method,
                "model": self.config.get("model") if method == "cross_encoder" else None,
                "private_reasoning": private_reasoning
            },
            execution_time_ms=execution_time
        )
    
    async def _rerank_cross_encoder(self, query: str, spans: List) -> List[tuple]:
        """Rerank using cross-encoder model"""
        from sentence_transformers import CrossEncoder
        
        # Lazy load model
        if self._reranker_model is None:
            model_name = self.config.get("model", "cross-encoder/ms-marco-MiniLM-L-6-v2")
            logger.info(f"Loading cross-encoder model: {model_name}")
            self._reranker_model = CrossEncoder(model_name)
        
        # Prepare query-document pairs
        pairs = [[query, span.text[:512]] for span in spans]
        
        # Score all pairs
        scores = self._reranker_model.predict(pairs)
        
        return list(zip(scores, spans))
    
    def _rerank_bm25(self, query: str, spans: List) -> List[tuple]:
        """Rerank using BM25"""
        from rank_bm25 import BM25Okapi
        
        # Tokenize documents
        tokenized_docs = [span.text.lower().split() for span in spans]
        
        # Initialize BM25
        bm25 = BM25Okapi(tokenized_docs)
        
        # Score documents
        tokenized_query = query.lower().split()
        scores = bm25.get_scores(tokenized_query)
        
        return list(zip(scores, spans))
    
    def _rerank_simple(self, spans: List) -> List[tuple]:
        """Simple score-based reranking"""
        scored_spans = []
        for i, span in enumerate(spans):
            # Recency bonus (earlier = higher)
            recency_score = 1.0 - (i / max(len(spans), 1))
            
            # Original score from metadata
            orig_score = span.doc_meta.get('score', 0.5) if span.doc_meta else 0.5
            
            # Combine scores
            final_score = orig_score * 0.7 + recency_score * 0.3
            
            scored_spans.append((final_score, span))
        
        return scored_spans

