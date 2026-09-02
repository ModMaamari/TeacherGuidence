"""
Answer drafter component for generating evidence-based answers.
"""

import time
from typing import Dict, Any
from loguru import logger

from agentsim.components.base import ComponentSpec, ComponentResult, ComponentRegistry
from agentsim.components.synthesis.base import SynthesisComponent
from agentsim.workflow.context import WorkflowContext
from agentsim.prompts import PromptManager


@ComponentRegistry.register("answer_drafter")
class AnswerDrafterComponent(SynthesisComponent):
    """
    Generates answer using only the collected evidence.
    
    This component uses an LLM to synthesize an answer from evidence
    WITHOUT using internal knowledge (evidence-only answers).
    """
    
    def __init__(self, config: Dict[str, Any] = None, llm_client=None):
        """
        Initialize answer drafter.
        
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
            name="answer_drafter",
            category=self.category,
            description="Generates answer from evidence only",
            input_keys=["query", "evidence_store"],
            output_keys=["draft_answer"],
            config_schema={
                "max_length": {
                    "type": "integer",
                    "default": 500,
                    "description": "Maximum answer length"
                },
                "temperature": {
                    "type": "number",
                    "default": 0.1,
                    "description": "LLM temperature (low for factual)"
                },
                "max_evidence": {
                    "type": "integer",
                    "default": 10,
                    "description": "Maximum evidence spans to use"
                }
            },
            requires_llm=True
        )
    
    async def execute(self, context: WorkflowContext) -> ComponentResult:
        """
        Execute answer generation.
        
        Args:
            context: Workflow context
            
        Returns:
            ComponentResult with generated answer
        """
        start_time = time.time()
        
        if not self.llm_client:
            return ComponentResult(
                success=False,
                error="LLM client not provided",
                execution_time_ms=0
            )
        
        if not context.evidence_store:
            return ComponentResult(
                success=False,
                error="No evidence available for synthesis",
                execution_time_ms=(time.time() - start_time) * 1000
            )
        
        prompt_data = self._prepare_prompt(context)
        prompt = prompt_data["prompt"]
        max_length = prompt_data["max_length"]
        temperature = prompt_data["temperature"]
        
        try:
            result = await self.llm_client.get_completion(
                prompt=prompt,
                temperature=temperature,
                max_tokens=max_length,
                return_usage=True
            )
            
            # Extract text and usage from result
            if isinstance(result, dict):
                response = result["text"]
                usage = result.get("usage", {})
                total_tokens = usage.get("total_tokens", 0)
            else:
                response = result
                total_tokens = 0
            
            # Extract reasoning and answer from LLM response
            private_reasoning = ""
            answer = ""
            
            if "REASONING:" in response and "ANSWER:" in response:
                parts = response.split("ANSWER:")
                private_reasoning = parts[0].replace("REASONING:", "").strip()
                answer = parts[1].strip()
            else:
                # Fallback if format not followed
                answer = response.strip()
                private_reasoning = f"Generated answer from {prompt_data['evidence_count']} evidence spans"
            
            if not answer:
                return ComponentResult(
                    success=False,
                    error="Empty answer generated",
                    execution_time_ms=(time.time() - start_time) * 1000
                )
            
            # Store in context metadata
            context.metadata["draft_answer"] = answer
            
            execution_time = (time.time() - start_time) * 1000
            evidence_count = prompt_data["evidence_count"]
            
            return ComponentResult(
                success=True,
                data={
                    "answer": answer,
                    "answer_length": len(answer),
                    "evidence_used": evidence_count
                },
                metadata={
                    "query": context.query,
                    "llm_input": prompt,
                    "llm_output": response,  # Store full response with reasoning
                    "private_reasoning": private_reasoning,
                    "tokens": total_tokens  # Add token count
                },
                execution_time_ms=execution_time
            )
            
        except Exception as e:
            logger.error(f"Answer drafter error: {e}")
            return ComponentResult(
                success=False,
                error=str(e),
                execution_time_ms=(time.time() - start_time) * 1000
            )
    
    def _prepare_prompt(self, context: WorkflowContext) -> Dict[str, Any]:
        """Build prompt components shared by teacher and consultants."""
        max_evidence = self.config.get("max_evidence", 10)
        temperature = self.config.get("temperature", 0.1)
        max_length = self.config.get("max_length", 500)
        
        evidence_spans = context.evidence_store[:max_evidence]
        evidence_text = "\n\n".join([
            f"[{i+1}] {span.text[:300]}"
            for i, span in enumerate(evidence_spans)
        ])
        
        prompt = self.prompt_manager.render(
            "synthesis/draft_answer",
            query=context.query,
            evidence_text=evidence_text
        )
        
        return {
            "prompt": prompt,
            "temperature": temperature,
            "max_length": max_length,
            "evidence_count": len(evidence_spans)
        }
    
    async def generate_consultant_output(self, context: WorkflowContext, model_config) -> Dict[str, Any]:
        """Generate consultant model answer without mutating context."""
        prompt_data = self._prepare_prompt(context)
        prompt = prompt_data["prompt"]
        max_length = prompt_data["max_length"]
        temperature = getattr(model_config, "temperature", self.config.get("temperature", 0.7))
        
        response = await self.llm_client.get_completion(
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_length,
            model=model_config.model_id
        )
        
        answer = response.strip() if response else ""
        
        return {
            "model": model_config.name,
            "model_id": model_config.model_id,
            "answer": answer,
            "answer_length": len(answer),
            "temperature": temperature
        }

