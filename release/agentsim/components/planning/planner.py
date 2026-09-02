"""
Planner component for query decomposition.
"""

import time
import json
from typing import Dict, Any, List
from loguru import logger

from agentsim.components.base import ComponentSpec, ComponentResult, ComponentRegistry
from agentsim.components.planning.base import PlanningComponent
from agentsim.workflow.context import WorkflowContext
from agentsim.prompts import PromptManager


@ComponentRegistry.register("planner")
class PlannerComponent(PlanningComponent):
    """
    Decomposes complex queries into focused sub-goals.
    
    This component uses an LLM to analyze the query and break it down
    into multiple specific search angles or sub-questions.
    """
    
    def __init__(self, config: Dict[str, Any] = None, llm_client=None):
        """
        Initialize planner.
        
        Args:
            config: Component configuration
            llm_client: LLM client for decomposition
        """
        super().__init__(config)
        self.llm_client = llm_client
        self.prompt_manager = PromptManager()
    
    @property
    def spec(self) -> ComponentSpec:
        return ComponentSpec(
            name="planner",
            category=self.category,
            description="Decomposes query into sub-goals",
            input_keys=["query"],
            output_keys=["subgoals"],
            config_schema={
                "max_subgoals": {
                    "type": "integer",
                    "default": 3,
                    "min": 1,
                    "max": 10,
                    "description": "Maximum number of sub-goals to generate"
                },
                "temperature": {
                    "type": "number",
                    "default": 0.7,
                    "min": 0.0,
                    "max": 1.0,
                    "description": "LLM temperature for creativity"
                }
            },
            requires_llm=True
        )
    
    async def execute(self, context: WorkflowContext) -> ComponentResult:
        """
        Execute query decomposition.
        
        Args:
            context: Workflow context
            
        Returns:
            ComponentResult with sub-goals
        """
        start_time = time.time()
        
        if not self.llm_client:
            return ComponentResult(
                success=False,
                error="LLM client not provided",
                execution_time_ms=0
            )
        
        max_subgoals = self.config.get("max_subgoals", 3)
        temperature = self.config.get("temperature", 0.7)
        
        # Load prompt from template
        prompt = self.prompt_manager.render(
            "planning/decompose",
            query=context.query,
            max_subgoals=max_subgoals
        )
        
        try:
            # Teacher model generates subgoals
            response = await self.llm_client.get_completion(
                prompt=prompt,
                temperature=temperature,
                max_tokens=300
            )
            if response is None:
                raise ValueError("LLM returned no content for sub-goal decomposition")
            
            logger.debug(f"Planner raw LLM response: {response}")
            
            # Extract reasoning and sub-goals from response
            private_reasoning = ""
            if "REASONING:" in response and "SUB-GOALS:" in response:
                parts = response.split("SUB-GOALS:")
                private_reasoning = parts[0].replace("REASONING:", "").strip()
                subgoals_text = parts[1].strip()
            else:
                # Fallback if format not followed
                subgoals_text = response
                private_reasoning = "Query decomposition performed"
            
            # Parse sub-goals from response
            subgoals = self._parse_subgoals(subgoals_text, max_subgoals)
            if not subgoals:
                snippet = subgoals_text.strip()[:200] if isinstance(subgoals_text, str) else str(subgoals_text)
                logger.error(f"Planner could not parse sub-goals from LLM response: {snippet}")
                raise ValueError(f"LLM response did not contain parseable sub-goals: {snippet}")
            
            # Get consultant suggestions if available
            consultant_subgoals = []
            if hasattr(context, 'consultant_models') and context.consultant_models:
                for consultant in context.consultant_models:
                    try:
                        consultant_response = await self.llm_client.get_completion(
                            prompt=prompt,
                            temperature=consultant.temperature,
                            max_tokens=300,
                            model=consultant.model_id
                        )
                        consultant_parsed = self._parse_subgoals(consultant_response, max_subgoals)
                        consultant_subgoals.append({
                            "model": consultant.name,
                            "model_id": consultant.model_id,
                            "subgoals": consultant_parsed
                        })
                        logger.info(f"Consultant {consultant.name} suggested {len(consultant_parsed)} subgoals")
                    except Exception as e:
                        logger.warning(f"Consultant {consultant.name} failed: {e}")
            
            # Merge unique consultant subgoals with teacher's
            all_subgoals = set(subgoals)
            for consultant_data in consultant_subgoals:
                for sg in consultant_data["subgoals"]:
                    if sg not in all_subgoals and len(all_subgoals) < max_subgoals * 2:
                        all_subgoals.add(sg)
                        subgoals.append(sg)
            
            # Store in context
            context.subgoals = subgoals
            
            execution_time = (time.time() - start_time) * 1000
            
            # Augment private reasoning with consultant info if present
            if consultant_subgoals:
                private_reasoning += (
                    f"\n\nConsultant models ({', '.join([c['model'] for c in consultant_subgoals])}) "
                    f"contributed additional perspectives, expanding coverage to {len(subgoals)} total subgoals."
                )
            
            return ComponentResult(
                success=True,
                data={
                    "subgoals": subgoals,
                    "count": len(subgoals),
                    "consultant_subgoals": consultant_subgoals if consultant_subgoals else None
                },
                metadata={
                    "query": context.query,
                    "max_subgoals": max_subgoals,
                    "llm_input": prompt,
                    "llm_output": response,
                    "parameters": {
                        "subgoals": subgoals,
                        "count": len(subgoals)
                    },
                    "private_reasoning": private_reasoning,
                    "rationale_tag": "BREADTH"
                },
                execution_time_ms=execution_time
            )
            
        except Exception as e:
            logger.error(f"Planner error: {e}")
            return ComponentResult(
                success=False,
                error=str(e),
                execution_time_ms=(time.time() - start_time) * 1000
            )
    
    def _parse_subgoals(self, response: str, max_count: int) -> List[str]:
        """
        Parse sub-goals from LLM response.
        
        Args:
            response: LLM response text
            max_count: Maximum number of sub-goals
            
        Returns:
            List of parsed sub-goals
        """
        subgoals = []
        lines = response.strip().split('\n')
        
        for line in lines:
            line = line.strip()
            # Match lines like "1. subgoal" or "- subgoal"
            if line and (line[0].isdigit() or line.startswith('-') or line.startswith('•')):
                # Remove numbering and clean up
                cleaned = line.lstrip('0123456789.-•)').strip()
                if cleaned and len(cleaned) > 5:  # Filter out too short
                    subgoals.append(cleaned)
        
        if not subgoals:
            raise ValueError("No sub-goals could be parsed from LLM response")
        
        return subgoals[:max_count]

