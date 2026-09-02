"""Adaptive Mode: Dynamic component selection with HATEOAS"""

from typing import Dict, Any, List, Optional
from loguru import logger

from agentsim.components.base import ComponentRegistry


class AdaptiveRunner:
    """Adaptive execution where teacher selects next components"""
    
    def __init__(self, template, clients, workflow_executor=None):
        self.template = template
        self.clients = clients
        self.workflow_executor = workflow_executor
        self.workflow = None  # Store workflow for component filtering
    
    async def run(
        self,
        workflow,
        dataset_sample: Dict[str, Any],
        initial_metadata: Dict[str, Any] = None,
        sample_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Execute mode 2 simulation
        
        Args:
            workflow: Workflow to execute
            dataset_sample: Sample with query and answer
            initial_metadata: Optional metadata to inject into context (e.g., knowledge base)
        """
        
        # Store workflow for component filtering
        self.workflow = workflow
        self.initial_metadata = initial_metadata or {}
        
        query = dataset_sample["query"]
        gold_answer = dataset_sample["answer"]
        
        # Track state
        context = None
        iteration = 0
        
        while iteration < self.template.max_iterations:
            iteration += 1
            logger.info(f"Iteration {iteration}: Asking teacher for next actions")
            
            # Get available components (HATEOAS)
            available = self._get_available_components(context)
            available_names = [c['name'] for c in available]
            logger.info(f"Available components (HATEOAS): {available_names}")
            
            # Ask teacher what to do next
            next_actions = await self._teacher_decision(
                query, gold_answer, context, available
            )
            
            if not next_actions or next_actions == ["finish"]:
                break
            
            # Execute selected components
            context = await self._execute_actions(next_actions, query, context)
            
            # Check if synthesis was generated
            synthesis = context.metadata.get("final_answer") or context.metadata.get("draft_answer")
            
            if synthesis:
                similarity = await self._compute_similarity(synthesis, gold_answer)
                
                if similarity >= self.template.similarity_threshold:
                    logger.info(f"✓ Threshold met after {iteration} iterations")
                    return {
                        "success": True,
                        "iterations": iteration,
                        "synthesis": synthesis,
                        "similarity": similarity,
                        "context": context
                    }
        
        # Max iterations reached or no synthesis
        synthesis = context.metadata.get("final_answer") or context.metadata.get("draft_answer") if context else None
        return {
            "success": False,
            "iterations": iteration,
            "synthesis": synthesis,
            "similarity": 0.0,
            "context": context
        }
    
    def _get_available_components(self, context) -> List[Dict[str, str]]:
        """Get components available based on current state (HATEOAS)"""
        available = []
        
        # Determine current state
        has_evidence = False
        evidence_count = 0
        
        if context and hasattr(context, 'evidence_store') and context.evidence_store:
            evidence_count = len(context.evidence_store)
            has_evidence = evidence_count > 0
        
        logger.debug(f"HATEOAS state check: has_evidence={has_evidence}, evidence_count={evidence_count}")
        
        # Get components from the workflow definition (only those in the workflow)
        if self.workflow and hasattr(self.workflow, 'components'):
            workflow_component_types = set()
            for comp_config in self.workflow.components:
                comp_type = comp_config.get("type")
                if comp_type:
                    workflow_component_types.add(comp_type)
        else:
            # Fallback to all registered components if workflow not available
            workflow_component_types = set(ComponentRegistry.list_components())
        
        # Filter by workflow and state
        for comp_name in workflow_component_types:
            spec = ComponentRegistry.get_spec(comp_name)
            if not spec:
                continue
            
            # Check if component can be used based on current state
            can_use = True
            reason = ""
            
            # Retrieval components: always available (they generate evidence)
            if spec.category.value == "retrieval":
                can_use = True
            
            # Planning components: always available (they generate queries/subgoals)
            elif spec.category.value == "planning":
                can_use = True
            
            # Processing components: require evidence
            elif spec.category.value == "processing":
                if not has_evidence:
                    can_use = False
                    reason = "requires evidence"
            
            # Synthesis components: require evidence
            elif spec.category.value == "synthesis":
                if not has_evidence:
                    can_use = False
                    reason = "requires evidence"
            
            # Verification components: require evidence
            elif spec.category.value == "verification":
                if not has_evidence:
                    can_use = False
                    reason = "requires evidence"
            
            if can_use:
                available.append({
                    "name": comp_name,
                    "category": spec.category.value,
                    "description": spec.description
                })
                logger.debug(f"  ✓ {comp_name} ({spec.category.value}) - available")
            else:
                logger.debug(f"  ✗ {comp_name} ({spec.category.value}) - not available ({reason})")
        
        return available
    
    async def _teacher_decision(
        self, query: str, gold: str, context, available: List[Dict]
    ) -> List[str]:
        """Ask teacher model what components to execute next"""
        
        llm = self.clients.get("llm_client")
        from agentsim.workflow.context import Message
        
        # Build detailed state summary
        has_evidence = context and hasattr(context, 'evidence_store') and len(context.evidence_store) > 0
        evidence_count = len(context.evidence_store) if has_evidence else 0
        
        if not context:
            state_summary = "No actions taken yet. No evidence retrieved."
        elif evidence_count > 0:
            state_summary = f"Evidence retrieved: {evidence_count} items available for processing/synthesis"
        else:
            state_summary = "Actions taken but no evidence retrieved yet"
        
        available_list = "\n".join([
            f"- {c['name']} ({c['category']}): {c['description']}" for c in available
        ])
        
        prompt = f"""You are planning the next actions for a RAG workflow.

Current state: {state_summary}
Query: {query}

AVAILABLE components (only these can be used now):
{available_list}

IMPORTANT CONSTRAINTS:
- You can ONLY choose from the available components listed above
- Components not listed are NOT available in the current state
- Processing/synthesis components require evidence to be retrieved first
- Choose components in a logical order (e.g., retrieve before process, process before synthesize)

Explain your reasoning first, then provide your decision.

Format:
REASONING: [Why you're choosing these components and what you hope to achieve]
DECISION: ["component1", "component2", ...]

Or DECISION: ["finish"] if you have a complete answer."""
        
        try:
            completion = await llm.get_completion(prompt, temperature=0.7, return_usage=True)
            if isinstance(completion, dict):
                response = completion.get("text", "")
                usage = completion.get("usage", {}) or {}
                tokens = usage.get("total_tokens", 0)
            else:
                response = str(completion)
                tokens = 0
            
            # Extract reasoning and decision
            import json
            reasoning = ""
            actions = []
            
            if "REASONING:" in response and "DECISION:" in response:
                parts = response.split("DECISION:")
                reasoning = parts[0].replace("REASONING:", "").strip()
                decision_text = parts[1].strip()
                
                # Extract JSON from decision
                start = decision_text.find('[')
                end = decision_text.rfind(']') + 1
                if start != -1 and end > start:
                    actions = json.loads(decision_text[start:end])
            else:
                # Fallback: try to parse as JSON
                actions = json.loads(response)
                reasoning = f"Selected {len(actions)} components for execution"
            
            # Store reasoning in context for trace
            if context:
                context.metadata["adaptive_reasoning"] = reasoning
                context.metadata["adaptive_actions"] = actions
                # Add a synthetic message to include teacher decision in traces and token accounting
                context.add_message(Message(
                    turn=max(len(context.messages) + 1, 1),
                    component="teacher_decision",
                    thought=reasoning,
                    action={"tool": "teacher_decision", "parameters": {}},
                    observation={"decision": actions},
                    verdict="PROCEED",
                    latency_ms=0.0,
                    execution_time_ms=0.0,
                    llm_input=prompt,
                    llm_output=response,
                    rationale_tag="",
                    stop_condition="CONTINUE",
                    tool_input={},
                    tool_output={"decision": actions},
                    tokens=tokens,
                    evidence_count=evidence_count
                ))
            
            logger.info(f"Teacher decision: {actions}")
            logger.debug(f"Teacher reasoning: {reasoning}")
            
            return actions
        except Exception as e:
            logger.error(f"Teacher decision error: {e}")
            return []
    
    async def _execute_actions(self, actions: List[str], query: str, prev_context):
        """Execute selected components dynamically"""
        from agentsim.workflow.context import WorkflowContext
        from agentsim.workflow.loader import WorkflowDefinition
        import time
        
        # Create or reuse context
        # On first iteration, inject initial_metadata (e.g., knowledge base)
        if prev_context is None:
            task_id = f"adaptive_{int(time.time())}"
            context = WorkflowContext(task_id=task_id, query=query, metadata=self.initial_metadata.copy())
            logger.debug(f"Created new context with initial_metadata: {list(self.initial_metadata.keys())}")
        else:
            context = prev_context
        
        # Build dynamic workflow from selected actions
        components = []
        for action in actions:
            if action != "finish":
                components.append({
                    "type": action,
                    "config": {}
                })
        
        # Create workflow definition
        workflow = WorkflowDefinition(
            id="adaptive_dynamic",
            name="Adaptive Dynamic Workflow",
            description="Dynamically generated workflow",
            reasoning_style="adaptive",
            components=components,
            config={}
        )
        
        # Execute workflow with existing context to preserve evidence
        if self.workflow_executor:
            try:
                context = await self.workflow_executor.execute(workflow, query, context=context)
                return context
            except Exception as e:
                logger.error(f"Workflow execution error: {e}")
                import traceback
                logger.error(traceback.format_exc())
                if prev_context:
                    return prev_context
                else:
                    task_id = f"adaptive_fallback_{int(time.time())}"
                    return WorkflowContext(task_id=task_id, query=query)
        else:
            logger.warning("No workflow executor available")
            if prev_context:
                return prev_context
            else:
                task_id = f"adaptive_noexec_{int(time.time())}"
                return WorkflowContext(task_id=task_id, query=query)
    
    async def _compute_similarity(self, synthesis: str, gold: str) -> float:
        """Compute similarity using local embeddings"""
        
        # Handle empty strings
        if not synthesis or not synthesis.strip():
            logger.warning("Synthesis is empty, returning similarity 0.0")
            return 0.0
        
        if not gold or not gold.strip():
            logger.warning("Gold answer is empty, returning similarity 1.0 (no comparison needed)")
            return 1.0  # No gold answer to compare against, so pass
        
        try:
            import numpy as np
            from sklearn.metrics.pairwise import cosine_similarity
            
            llm = self.clients.get("llm_client")
            if not llm:
                return 0.0
            
            emb1 = await llm.get_embedding(synthesis)
            emb2 = await llm.get_embedding(gold)
            
            sim = cosine_similarity([emb1], [emb2])[0][0]
            return float(sim)
        except Exception as e:
            logger.error(f"Similarity computation error: {e}")
            return 0.0

