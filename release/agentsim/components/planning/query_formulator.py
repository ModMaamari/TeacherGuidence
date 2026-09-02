"""
Query formulator component for generating search queries.
"""

import time
import json
from typing import Dict, Any
from loguru import logger

from agentsim.components.base import ComponentSpec, ComponentResult, ComponentRegistry
from agentsim.components.planning.base import PlanningComponent
from agentsim.workflow.context import WorkflowContext
from agentsim.prompts import PromptManager


@ComponentRegistry.register("query_formulator")
class QueryFormulatorComponent(PlanningComponent):
    """
    Generates effective search queries for sub-goals.
    
    This component takes sub-goals from the planner and creates
    optimized search queries for each one.
    """
    
    def __init__(self, config: Dict[str, Any] = None, llm_client=None):
        """
        Initialize query formulator.
        
        Args:
            config: Component configuration
            llm_client: LLM client
        """
        super().__init__(config)
        self.llm_client = llm_client
        self.prompt_manager = PromptManager()
    
    @property
    def spec(self) -> ComponentSpec:
        return ComponentSpec(
            name="query_formulator",
            category=self.category,
            description="Generates search queries for sub-goals",
            input_keys=["subgoals"],
            output_keys=["queries"],
            config_schema={
                "queries_per_subgoal": {
                    "type": "integer",
                    "default": 2,
                    "min": 1,
                    "max": 5,
                    "description": "Number of query variants per sub-goal"
                },
                "temperature": {
                    "type": "number",
                    "default": 0.8,
                    "min": 0.0,
                    "max": 1.0,
                    "description": "LLM temperature"
                }
            },
            requires_llm=True
        )
    
    async def execute(self, context: WorkflowContext) -> ComponentResult:
        """
        Execute query formulation.
        
        Args:
            context: Workflow context
            
        Returns:
            ComponentResult with queries
        """
        start_time = time.time()
        
        if not self.llm_client:
            return ComponentResult(
                success=False,
                error="LLM client not provided",
                execution_time_ms=0
            )
        
        if not context.subgoals:
            # If no subgoals, use the main query
            context.queries.append(context.query)
            return ComponentResult(
                success=True,
                data={"queries": [context.query], "fallback": True},
                execution_time_ms=(time.time() - start_time) * 1000
            )
        
        queries_per_subgoal = self.config.get("queries_per_subgoal", 2)
        temperature = self.config.get("temperature", 0.8)
        
        # Load prompt from template
        prompt = self.prompt_manager.render(
            "planning/formulate_query",
            subgoals_json=json.dumps(context.subgoals, indent=2),
            queries_per_subgoal=queries_per_subgoal
        )
        
        try:
            response = await self.llm_client.get_completion(
                prompt=prompt,
                temperature=temperature,
                max_tokens=400
            )
            
            # Parse JSON response
            result = json.loads(response)
            queries = result.get("queries", [])
            
            # Validate queries
            queries = [q for q in queries if isinstance(q, str) and q.strip()]
            
            if not queries:
                # Fallback: use subgoals as queries
                queries = context.subgoals
            
            # Add to context
            context.queries.extend(queries)
            
            execution_time = (time.time() - start_time) * 1000
            
            return ComponentResult(
                success=True,
                data={"queries": queries, "count": len(queries)},
                metadata={"subgoals": context.subgoals},
                execution_time_ms=execution_time
            )
            
        except Exception as e:
            logger.error(f"Query formulator error: {e}")
            
            # Fallback: use subgoals as queries
            queries = context.subgoals
            context.queries.extend(queries)
            
            return ComponentResult(
                success=True,
                data={"queries": queries, "fallback_used": True},
                metadata={"error": str(e)},
                execution_time_ms=(time.time() - start_time) * 1000
            )

