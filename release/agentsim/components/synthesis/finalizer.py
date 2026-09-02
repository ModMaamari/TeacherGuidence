"""
Finalizer component for formatting final output.
"""

import time
from typing import Dict, Any
from loguru import logger

from agentsim.components.base import ComponentSpec, ComponentResult, ComponentRegistry
from agentsim.components.synthesis.base import SynthesisComponent
from agentsim.workflow.context import WorkflowContext


@ComponentRegistry.register("finalizer")
class FinalizerComponent(SynthesisComponent):
    """
    Formats the final answer with citations and metadata.
    """
    
    @property
    def spec(self) -> ComponentSpec:
        return ComponentSpec(
            name="finalizer",
            category=self.category,
            description="Formats final answer with citations",
            input_keys=["draft_answer", "evidence_store"],
            output_keys=["final_answer"],
            config_schema={
                "format": {
                    "type": "string",
                    "default": "markdown",
                    "enum": ["markdown", "plain"],
                    "description": "Output format"
                },
                "include_citations": {
                    "type": "boolean",
                    "default": True,
                    "description": "Include citation list"
                }
            },
            requires_llm=False
        )
    
    async def execute(self, context: WorkflowContext) -> ComponentResult:
        """
        Execute finalization.
        
        Args:
            context: Workflow context
            
        Returns:
            ComponentResult with final answer and separate citations
        """
        start_time = time.time()
        
        draft_answer = context.metadata.get("draft_answer")
        if not draft_answer:
            return ComponentResult(
                success=False,
                error="No draft answer available",
                execution_time_ms=(time.time() - start_time) * 1000
            )
        
        # Answer is ONLY the draft text, no citations embedded
        final_answer = draft_answer.strip()
        
        # Build citations as separate metadata
        citations = []
        if context.evidence_store:
            for i, span in enumerate(context.evidence_store[:10], 1):
                citations.append({
                    "id": i,
                    "doc_id": span.id,
                    "source": span.source,
                    "text": span.text[:200],
                    "score": span.doc_meta.get('score', 0.0) if span.doc_meta else 0.0
                })
        
        # Store in context
        context.metadata["final_answer"] = final_answer
        context.metadata["citations"] = citations
        
        execution_time = (time.time() - start_time) * 1000
        
        logger.info(f"Finalized answer: {len(final_answer)} chars, {len(citations)} citations")
        
        return ComponentResult(
            success=True,
            data={
                "final_answer": final_answer,
                "answer_length": len(final_answer),
                "citations_count": len(citations)
            },
            metadata={
                "citations": citations
            },
            execution_time_ms=execution_time
        )

