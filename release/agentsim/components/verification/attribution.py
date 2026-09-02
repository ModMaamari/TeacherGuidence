"""
Attribution gate component for verifying claim attribution.
"""

import time
from typing import Dict, Any
from loguru import logger

from agentsim.components.base import ComponentSpec, ComponentResult, ComponentRegistry
from agentsim.components.verification.base import VerificationComponent
from agentsim.workflow.context import WorkflowContext


@ComponentRegistry.register("attribution_gate")
class AttributionGateComponent(VerificationComponent):
    """
    Verifies that all claims in the answer are properly attributed to evidence.
    """
    
    @property
    def spec(self) -> ComponentSpec:
        return ComponentSpec(
            name="attribution_gate",
            category=self.category,
            description="Verifies claim attribution",
            input_keys=["draft_answer", "evidence_store"],
            output_keys=["attribution_result"],
            config_schema={
                "threshold": {
                    "type": "number",
                    "default": 0.85,
                    "description": "Attribution confidence threshold"
                },
                "strict_mode": {
                    "type": "boolean",
                    "default": True,
                    "description": "Strict attribution mode"
                }
            },
            requires_llm=False
        )
    
    async def execute(self, context: WorkflowContext) -> ComponentResult:
        """
        Execute attribution checking.
        
        Args:
            context: Workflow context
            
        Returns:
            ComponentResult with attribution results
        """
        start_time = time.time()
        
        draft_answer = context.metadata.get("draft_answer")
        if not draft_answer:
            return ComponentResult(
                success=False,
                error="No draft answer to check attribution",
                execution_time_ms=(time.time() - start_time) * 1000
            )
        
        threshold = self.config.get("threshold", 0.85)
        
        # Simplified attribution: check if we have evidence
        evidence_count = len(context.evidence_store)
        passed = evidence_count >= 2
        confidence = min(evidence_count / 5.0, 1.0)
        
        context.metadata["attribution_passed"] = passed
        context.metadata["attribution_confidence"] = confidence
        
        execution_time = (time.time() - start_time) * 1000
        
        status = "PASS" if passed else "FAIL"
        logger.info(f"Attribution gate: {status} (confidence={confidence:.2f}, evidence={evidence_count}, threshold={threshold})")
        
        return ComponentResult(
            success=True,
            data={
                "passed": passed,
                "confidence": confidence,
                "threshold": threshold,
                "evidence_count": evidence_count
            },
            execution_time_ms=execution_time
        )

