"""
Filter component for quality-based filtering.
"""

import time
from typing import Dict, Any
from loguru import logger

from agentsim.components.base import ComponentSpec, ComponentResult, ComponentRegistry
from agentsim.components.processing.base import ProcessingComponent
from agentsim.workflow.context import WorkflowContext


@ComponentRegistry.register("filter")
class FilterComponent(ProcessingComponent):
    """
    Filters evidence based on quality criteria (length, score, etc.).
    """
    
    @property
    def spec(self) -> ComponentSpec:
        return ComponentSpec(
            name="filter",
            category=self.category,
            description="Filters evidence by quality criteria",
            input_keys=["evidence_store"],
            output_keys=["evidence_store"],
            config_schema={
                "min_length": {
                    "type": "integer",
                    "default": 30,
                    "description": "Minimum text length"
                },
                "max_length": {
                    "type": "integer",
                    "default": 10000,
                    "description": "Maximum text length"
                },
                "min_score": {
                    "type": "number",
                    "default": 0.0,
                    "description": "Minimum relevance score"
                },
                "max_results": {
                    "type": "integer",
                    "default": 15,
                    "description": "Maximum results to keep"
                }
            },
            requires_llm=False
        )
    
    async def execute(self, context: WorkflowContext) -> ComponentResult:
        """
        Execute filtering.
        
        Args:
            context: Workflow context
            
        Returns:
            ComponentResult with filtering statistics
        """
        start_time = time.time()
        
        original_count = len(context.evidence_store)
        min_length = self.config.get("min_length", 30)
        max_length = self.config.get("max_length", 10000)
        min_score = self.config.get("min_score", 0.0)
        max_results = self.config.get("max_results", 15)
        
        # Filter evidence
        filtered_spans = []
        for span in context.evidence_store:
            # Length check
            text_length = len(span.text)
            if text_length < min_length or text_length > max_length:
                continue
            
            # Score check
            score = span.doc_meta.get('score', 1.0) if span.doc_meta else 1.0
            if score < min_score:
                continue
            
            filtered_spans.append(span)
        
        # Limit to max_results
        context.evidence_store = filtered_spans[:max_results]
        
        execution_time = (time.time() - start_time) * 1000
        
        return ComponentResult(
            success=True,
            data={
                "original_count": original_count,
                "filtered_count": len(context.evidence_store),
                "removed": original_count - len(context.evidence_store)
            },
            metadata={
                "min_length": min_length,
                "max_length": max_length,
                "min_score": min_score
            },
            execution_time_ms=execution_time
        )

