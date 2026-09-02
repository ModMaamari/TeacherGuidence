"""
Evidence extractor component for selecting key spans.
"""

import time
from typing import Dict, Any
from loguru import logger

from agentsim.components.base import ComponentSpec, ComponentResult, ComponentRegistry
from agentsim.components.processing.base import ProcessingComponent
from agentsim.workflow.context import WorkflowContext


@ComponentRegistry.register("extractor")
class ExtractorComponent(ProcessingComponent):
    """
    Extracts the most relevant evidence spans for the query and sub-goals.
    """
    
    @property
    def spec(self) -> ComponentSpec:
        return ComponentSpec(
            name="extractor",
            category=self.category,
            description="Extracts key evidence spans",
            input_keys=["evidence_store", "subgoals"],
            output_keys=["evidence_store"],
            config_schema={
                "max_spans": {
                    "type": "integer",
                    "default": 20,
                    "description": "Maximum number of spans to extract"
                },
                "span_window": {
                    "type": "integer",
                    "default": 200,
                    "description": "Window size for span extraction"
                }
            },
            requires_llm=False
        )
    
    async def execute(self, context: WorkflowContext) -> ComponentResult:
        """
        Execute evidence extraction.
        
        Args:
            context: Workflow context
            
        Returns:
            ComponentResult with extraction statistics
        """
        start_time = time.time()
        
        original_count = len(context.evidence_store)
        max_spans = self.config.get("max_spans", 20)
        
        # Simple extraction: keep top N spans
        # In a more sophisticated version, this could use LLM to select
        # spans that best match sub-goals
        context.evidence_store = context.evidence_store[:max_spans]
        
        execution_time = (time.time() - start_time) * 1000
        
        extracted = len(context.evidence_store)
        logger.info(f"Extracted: {original_count} -> {extracted} spans (max={max_spans})")
        
        return ComponentResult(
            success=True,
            data={
                "original_count": original_count,
                "extracted_count": extracted,
                "max_spans": max_spans
            },
            execution_time_ms=execution_time
        )

