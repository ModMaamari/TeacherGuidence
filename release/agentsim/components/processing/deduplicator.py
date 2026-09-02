"""
Deduplicator component for removing duplicate evidence.
"""

import time
from typing import Dict, Any
from loguru import logger

from agentsim.components.base import ComponentSpec, ComponentResult, ComponentRegistry
from agentsim.components.processing.base import ProcessingComponent
from agentsim.workflow.context import WorkflowContext


@ComponentRegistry.register("deduplicator")
class DeduplicatorComponent(ProcessingComponent):
    """
    Removes duplicate evidence spans based on text similarity.
    """
    
    @property
    def spec(self) -> ComponentSpec:
        return ComponentSpec(
            name="deduplicator",
            category=self.category,
            description="Removes duplicate evidence",
            input_keys=["evidence_store"],
            output_keys=["evidence_store"],
            config_schema={
                "method": {
                    "type": "string",
                    "default": "text_prefix",
                    "enum": ["text_prefix", "exact"],
                    "description": "Deduplication method"
                },
                "prefix_length": {
                    "type": "integer",
                    "default": 50,
                    "description": "Text prefix length for comparison"
                },
                "max_results": {
                    "type": "integer",
                    "default": 20,
                    "description": "Maximum results to keep"
                }
            },
            requires_llm=False
        )
    
    async def execute(self, context: WorkflowContext) -> ComponentResult:
        """
        Execute deduplication.
        
        Args:
            context: Workflow context
            
        Returns:
            ComponentResult with deduplication statistics
        """
        start_time = time.time()
        
        original_count = len(context.evidence_store)
        method = self.config.get("method", "text_prefix")
        prefix_length = self.config.get("prefix_length", 50)
        max_results = self.config.get("max_results", 20)
        
        # Deduplicate
        seen = set()
        unique_spans = []
        
        for span in context.evidence_store:
            if method == "exact":
                key = span.text
            else:  # text_prefix
                key = span.text[:prefix_length]
            
            if key not in seen:
                seen.add(key)
                unique_spans.append(span)
        
        # Limit to max_results
        context.evidence_store = unique_spans[:max_results]
        
        execution_time = (time.time() - start_time) * 1000
        
        removed = original_count - len(context.evidence_store)
        logger.info(f"Deduplicated: {original_count} -> {len(context.evidence_store)} spans ({removed} removed, method={method})")
        
        return ComponentResult(
            success=True,
            data={
                "original_count": original_count,
                "unique_count": len(context.evidence_store),
                "removed": removed
            },
            metadata={"method": method},
            execution_time_ms=execution_time
        )

