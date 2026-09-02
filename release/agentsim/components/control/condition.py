"""
Condition component for conditional branching.
"""

import time
from typing import Dict, Any
from loguru import logger

from agentsim.components.base import ComponentSpec, ComponentResult, ComponentRegistry
from agentsim.components.control.base import ControlComponent
from agentsim.workflow.context import WorkflowContext


@ComponentRegistry.register("condition")
class ConditionComponent(ControlComponent):
    """
    Evaluates conditions and enables conditional workflow branching.
    """
    
    @property
    def spec(self) -> ComponentSpec:
        return ComponentSpec(
            name="condition",
            category=self.category,
            description="Conditional branching based on context state",
            input_keys=["metadata"],
            output_keys=["branch_taken"],
            config_schema={
                "condition_type": {
                    "type": "string",
                    "default": "threshold",
                    "enum": ["threshold", "evidence_count", "iteration_count"],
                    "description": "Type of condition to evaluate"
                },
                "threshold": {
                    "type": "number",
                    "default": 0.6,
                    "description": "Threshold value for comparison"
                },
                "on_true": {
                    "type": "string",
                    "default": "continue",
                    "description": "Action when condition is true"
                },
                "on_false": {
                    "type": "string",
                    "default": "continue",
                    "description": "Action when condition is false"
                }
            },
            requires_llm=False
        )
    
    async def execute(self, context: WorkflowContext) -> ComponentResult:
        """
        Execute condition evaluation.
        
        Args:
            context: Workflow context
            
        Returns:
            ComponentResult with branch decision
        """
        start_time = time.time()
        
        condition_type = self.config.get("condition_type", "threshold")
        threshold = self.config.get("threshold", 0.6)
        on_true = self.config.get("on_true", "continue")
        on_false = self.config.get("on_false", "continue")
        
        # Evaluate condition
        condition_met = False
        condition_value = 0.0
        
        if condition_type == "threshold":
            # Check some score in metadata
            condition_value = context.metadata.get("last_score", 0.0)
            condition_met = condition_value >= threshold
        elif condition_type == "evidence_count":
            # Check evidence count
            condition_value = len(context.evidence_store)
            condition_met = condition_value >= threshold
        elif condition_type == "iteration_count":
            # Check iteration count
            condition_value = context.metadata.get("iteration", 1)
            condition_met = condition_value <= threshold
        
        branch_taken = on_true if condition_met else on_false
        
        # Store decision in metadata
        context.metadata["last_condition_met"] = condition_met
        context.metadata["last_branch_taken"] = branch_taken
        
        execution_time = (time.time() - start_time) * 1000
        
        return ComponentResult(
            success=True,
            data={
                "condition_type": condition_type,
                "condition_value": condition_value,
                "threshold": threshold,
                "condition_met": condition_met,
                "branch_taken": branch_taken
            },
            execution_time_ms=execution_time
        )

