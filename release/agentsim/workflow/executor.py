"""
Workflow executor for running component pipelines.

This module executes workflows by sequentially running components
and managing the shared context.
"""

import time
from typing import Dict, Any, Optional, List
from loguru import logger

from agentsim.workflow.context import WorkflowContext, Message
from agentsim.workflow.loader import WorkflowDefinition
from agentsim.components.base import ComponentRegistry, ComponentResult
from datetime import datetime


class WorkflowExecutor:
    """
    Executes workflows by running components in sequence.
    
    The executor:
    1. Creates a WorkflowContext
    2. Instantiates components from the workflow definition
    3. Runs components in sequence
    4. Manages error handling and verdicts
    5. Returns the final context
    """
    
    def __init__(
        self,
        llm_client=None,
        opensearch_client=None,
        vector_client=None,
        chatnoir_client=None,
        retrieval_config=None,
        trace_exporter=None,
        parallel_consultants=None
    ):
        """
        Initialize workflow executor.
        
        Args:
            llm_client: LLM client for components that require it
            opensearch_client: OpenSearch client
            vector_client: Vector search client
            chatnoir_client: ChatNoir client (optional)
            retrieval_config: Retrieval configuration from simulation template
            trace_exporter: TraceExporter for streaming traces during execution
            parallel_consultants: Mapping of component_type to consultant model configs
        """
        self.llm_client = llm_client
        self.opensearch_client = opensearch_client
        self.vector_client = vector_client
        self.chatnoir_client = chatnoir_client
        self.retrieval_config = retrieval_config
        self.trace_exporter = trace_exporter
        self.parallel_consultants = parallel_consultants or {}
    
    async def execute(
        self,
        workflow: WorkflowDefinition,
        query: str,
        task_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        max_steps: Optional[int] = None,
        context: Optional[WorkflowContext] = None
    ) -> WorkflowContext:
        """
        Execute a workflow.
        
        Args:
            workflow: Workflow definition to execute
            query: User query
            task_id: Optional task identifier
            metadata: Optional metadata
            max_steps: Maximum number of steps (overrides workflow config)
            context: Optional existing context to reuse (preserves evidence)
            
        Returns:
            WorkflowContext with execution results
        """
        # Create or reuse context
        if context is None:
            # Generate task ID if not provided
            if task_id is None:
                task_id = f"{workflow.id}_{int(time.time())}"
            
            # Create new context
            context = WorkflowContext(
                task_id=task_id,
                query=query,
                metadata=metadata or {}
            )
        else:
            # Reuse existing context but update metadata
            if metadata:
                context.metadata.update(metadata)
        
        # Add workflow config to metadata
        context.metadata.update(workflow.config)
        
        logger.info(f"Starting workflow execution: {workflow.name} (id={context.task_id})")
        logger.info(f"Query: {query}")
        
        # Log if reusing context with existing evidence
        if context.evidence_store:
            logger.info(f"Reusing context with {len(context.evidence_store)} existing evidence items")
        
        # Determine max steps
        if max_steps is None:
            max_steps = workflow.config.get("max_steps", len(workflow.components))
        
        # Execute components
        start_time = time.time()
        
        for step_idx, comp_def in enumerate(workflow.components[:max_steps], start=1):
            comp_type = comp_def.get("type")
            comp_config = comp_def.get("config", {})
            
            logger.info(f"Step {step_idx}/{max_steps}: Executing {comp_type}")
            
            try:
                # Instantiate component
                component = self._instantiate_component(comp_type, comp_config)
                
                # Execute component
                step_start = time.time()
                result = await component.execute(context)
                step_time = (time.time() - step_start) * 1000
                
                # Create message for trace (pass context for evidence count fallback)
                message = self._create_message(
                    turn=step_idx,
                    component=comp_type,
                    result=result,
                    step_time=step_time,
                    context=context
                )
                context.add_message(message)
                
                # Consultant models (parallel answers)
                consultant_outputs = await self._run_consultant_models(comp_type, component, context)
                if consultant_outputs:
                    context.metadata.setdefault("consultant_outputs", {}).setdefault(comp_type, []).extend(consultant_outputs)
                    tool_output = message.tool_output if isinstance(message.tool_output, dict) else {}
                    tool_output["consultant_outputs"] = consultant_outputs
                    message.tool_output = tool_output
                
                # Stream trace entry if exporter is available
                if self.trace_exporter:
                    self._stream_trace_entry(message, context)
                
                # Log result
                if result.success:
                    logger.info(f"✓ {comp_type} completed in {step_time:.0f}ms")
                    logger.debug(f"  Data: {result.data}")
                else:
                    logger.warning(f"✗ {comp_type} failed: {result.error}")
                
                # Handle verdict
                if not result.success and comp_config.get("stop_on_error", False):
                    logger.warning(f"Stopping workflow due to error in {comp_type}")
                    break
                
                # Increment step counter
                context.increment_step()

                # Allow any component to terminate the workflow early by returning FINISH.
                if message.stop_condition == "FINISH":
                    logger.info(f"Stopping workflow because {comp_type} returned FINISH")
                    break

            except Exception as e:
                logger.error(f"Error executing component {comp_type}: {e}", exc_info=True)
                
                # Create error message
                message = Message(
                    turn=step_idx,
                    component=comp_type,
                    thought=f"Error: {str(e)}",
                    action={"tool": comp_type, "error": True},
                    observation={"error": str(e)},
                    verdict="FAIL",
                    latency_ms=0
                )
                context.add_message(message)
                
                # Stop on error if configured
                if comp_config.get("stop_on_error", False):
                    break
        
        total_time = (time.time() - start_time) * 1000
        
        logger.info(f"Workflow completed in {total_time:.0f}ms")
        logger.info(f"  Steps: {len(context.messages)}")
        logger.info(f"  Evidence: {len(context.evidence_store)}")
        logger.info(f"  Final answer: {context.metadata.get('final_answer', 'N/A')[:100]}")
        
        # Add timing to metadata
        context.metadata["total_execution_time_ms"] = total_time
        context.metadata["workflow_id"] = workflow.id
        context.metadata["workflow_name"] = workflow.name
        
        return context
    
    def _instantiate_component(self, comp_type: str, comp_config: Dict[str, Any]):
        """
        Instantiate a component with appropriate clients.
        
        Args:
            comp_type: Component type
            comp_config: Component configuration
            
        Returns:
            Instantiated component
        """
        # Get component class
        component = ComponentRegistry.get(comp_type, config=comp_config)
        
        # Inject clients based on requirements
        spec = component.spec
        
        # Inject LLM client if required
        if spec.requires_llm and hasattr(component, 'llm_client'):
            component.llm_client = self.llm_client
        
        # Inject retrieval clients
        if hasattr(component, 'opensearch_client'):
            component.opensearch_client = self.opensearch_client
        
        if hasattr(component, 'vector_client'):
            component.vector_client = self.vector_client
        
        if hasattr(component, 'chatnoir_client'):
            component.chatnoir_client = self.chatnoir_client
        
        return component
    
    async def _run_consultant_models(self, component_type: str, component, context):
        """Run consultant models for parallel outputs."""
        configs = self.parallel_consultants.get(component_type)
        if not configs:
            return []
        
        if not hasattr(component, "generate_consultant_output"):
            logger.debug(f"Component {component_type} does not support consultant outputs.")
            return []
        
        consultant_outputs = []
        for cfg in configs:
            try:
                output = await component.generate_consultant_output(context, cfg)
                if output:
                    consultant_outputs.append(output)
            except Exception as exc:
                logger.error(f"Consultant model {cfg.name} failed: {exc}")
                consultant_outputs.append({
                    "model": cfg.name,
                    "model_id": cfg.model_id,
                    "error": str(exc)
                })
        return consultant_outputs
    
    def _create_message(
        self,
        turn: int,
        component: str,
        result: ComponentResult,
        step_time: float,
        context: Any = None
    ) -> Message:
        """
        Create a message from component result.
        
        Args:
            turn: Turn number
            component: Component name
            result: Component result
            step_time: Execution time
            context: Optional workflow context for evidence count fallback
            
        Returns:
            Message
        """
        # Determine verdict
        if result.success:
            verdict = result.data.get("verdict", "PROCEED")
        else:
            verdict = "FAIL"
        
        # Extract evidence count from various sources
        evidence_count = 0
        if "retrieved_count" in result.data:
            # Retrieval components
            evidence_count = result.data["retrieved_count"]
        elif "filtered_count" in result.data:
            # Filter component
            evidence_count = result.data["filtered_count"]
        elif "reranked_count" in result.data:
            # Reranker component
            evidence_count = result.data["reranked_count"]
        elif "evidence_count" in result.data:
            # General evidence count
            evidence_count = result.data["evidence_count"]
        elif "subgoals" in result.data:
            # Planning components
            evidence_count = len(result.data["subgoals"])
        elif context and hasattr(context, 'evidence_store'):
            # Fallback: use actual evidence store length
            evidence_count = len(context.evidence_store)
        
        # Use private_reasoning from metadata if available, otherwise use thought
        thought_text = result.metadata.get("private_reasoning") or result.metadata.get("thought", f"Executed {component}")
        
        return Message(
            turn=turn,
            component=component,
            thought=thought_text,
            action={
                "tool": component,
                "parameters": result.metadata.get("parameters", {})
            },
            observation=result.data,
            verdict=verdict,
            latency_ms=step_time,
            execution_time_ms=step_time,
            # Extract rich trace data from metadata
            llm_input=result.metadata.get("llm_input"),
            llm_output=result.metadata.get("llm_output"),
            rationale_tag=result.metadata.get("rationale_tag", ""),
            stop_condition=(
                "ERROR" if not result.success
                else "FINISH" if verdict == "FINISH"
                else "CONTINUE"
            ),
            # Expose component metadata (e.g., retriever queries/k/corpus) to traces
            tool_input=result.metadata or {},
            tool_output=result.data,
            tokens=result.metadata.get("tokens", 0),
            evidence_count=evidence_count,
            error=result.error if not result.success else None
        )
    
    async def execute_iterative(
        self,
        workflow: WorkflowDefinition,
        query: str,
        expected_answer: Optional[str] = None,
        max_iterations: int = 3,
        answer_threshold: float = 0.6,
        task_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> WorkflowContext:
        """
        Execute workflow iteratively until answer matches or max iterations.
        
        This is useful for training data generation where you want to
        iterate until the workflow produces a satisfactory answer.
        
        Args:
            workflow: Workflow definition
            query: User query
            expected_answer: Expected answer for comparison
            max_iterations: Maximum iterations
            answer_threshold: Similarity threshold for answer matching
            task_id: Optional task identifier
            metadata: Optional metadata
            
        Returns:
            WorkflowContext with execution results
        """
        if task_id is None:
            task_id = f"{workflow.id}_iterative_{int(time.time())}"
        
        metadata = metadata or {}
        metadata["max_iterations"] = max_iterations
        metadata["answer_threshold"] = answer_threshold
        metadata["expected_answer"] = expected_answer
        
        logger.info(f"Starting iterative execution: max_iterations={max_iterations}")
        
        context = None
        
        for iteration in range(1, max_iterations + 1):
            logger.info(f"=== Iteration {iteration}/{max_iterations} ===")
            
            # Update metadata with iteration
            metadata["iteration"] = iteration
            
            # Execute workflow
            context = await self.execute(
                workflow=workflow,
                query=query,
                task_id=task_id,
                metadata=metadata
            )
            
            # Check if we have an answer
            final_answer = context.metadata.get("final_answer")
            
            if final_answer and expected_answer:
                # TODO: Implement answer matching logic
                # For now, just log
                logger.info(f"Generated answer: {final_answer[:100]}")
                logger.info(f"Expected answer: {expected_answer[:100]}")
                
                # Simple check: if answer is present, consider it done
                if final_answer:
                    logger.info(f"✓ Answer generated after {iteration} iterations")
                    break
            elif final_answer:
                logger.info(f"✓ Answer generated (no expected answer to compare)")
                break
            else:
                logger.warning(f"✗ No answer generated in iteration {iteration}")
        
        return context
    
    def _stream_trace_entry(self, message: Message, context: WorkflowContext):
        """
        Stream a trace entry to the trace exporter.
        
        Args:
            message: Message object from workflow execution
            context: Current workflow context
        """
        # Convert message to trace entry format
        # Extract private reasoning from observation if available
        private_reasoning = message.thought or ""
        if isinstance(message.observation, dict) and "private_reasoning" in message.observation:
            private_reasoning = message.observation["private_reasoning"]
        
        trace_entry = {
            "step_id": message.turn,
            "goal": message.thought or "",
            "action": {
                "tool": message.component,
                "parameters": message.action.get("parameters", {})
            },
            "rationale_tag": message.rationale_tag or "",
            "operator_intent": "",  # TODO: Extract from component
            "stop_condition": message.stop_condition or "CONTINUE",
            "timestamp": datetime.now().timestamp(),
            "private_reasoning": private_reasoning,
            "llm_input": message.llm_input,
            "llm_output": message.llm_output,
            "tool_input": message.tool_input,
            "tool_output": message.tool_output,
            "evidence_retrieved": message.citations if message.citations else None,
            "evidence_count": message.evidence_count,
            "execution_time_ms": message.execution_time_ms,
            "error": message.error
        }
        
        # Stream to trace file
        self.trace_exporter.stream_trace_entry(trace_entry)
        
        # Stream supervised entry if LLM was used
        if message.llm_input and message.llm_output:
            supervised_entry = {
                "step_id": message.turn,
                "input": message.llm_input,
                "output": message.llm_output,
                "tool": message.component,
                "rationale_tag": message.rationale_tag or "",
                "decision_label": message.stop_condition or "CONTINUE",
                "latency_ms": message.execution_time_ms,
                "tokens": message.tokens
            }
            self.trace_exporter.stream_supervised_entry(supervised_entry)
        
        # Stream trajectory entry
        trajectory_entry = {
            "state": {
                "step_id": message.turn,
                "query": context.query,
                "evidence_count": message.evidence_count,
                "evidence_available_pct": 0.0,  # TODO: Calculate
                "state_stage": self._determine_stage(message.component),
                "previous_tools": [m.component for m in context.messages[:-1]],
                "knowledge_base_size": context.metadata.get("knowledge_base_size", 0)
            },
            "action": {
                "tool": message.component,
                "parameters": message.action.get("parameters", {}),
                "rationale_tag": message.rationale_tag or ""
            },
            "reward": self._calculate_reward(message),
            "next_state": {},  # Will be filled in next step
            "done": message.stop_condition == "FINISH"
        }
        self.trace_exporter.stream_trajectory_entry(trajectory_entry)
    
    def _determine_stage(self, component: str) -> str:
        """Determine workflow stage from component"""
        if component in ["planner", "query_formulator"]:
            return "planning"
        elif component in ["chatnoir_retriever", "opensearch_retriever", "vector_retriever"]:
            return "retrieving"
        elif component in ["reranker", "deduplicator", "filter", "extractor"]:
            return "processing"
        elif component in ["answer_drafter", "finalizer"]:
            return "synthesizing"
        elif component in ["fact_checker", "attribution_gate"]:
            return "verifying"
        else:
            return "unknown"
    
    def _calculate_reward(self, message: Message) -> float:
        """Calculate reward for RL training"""
        reward = 0.0
        
        # Positive reward for successful execution
        if not message.error:
            reward += 0.1
        
        # Reward for retrieving evidence
        if message.evidence_count > 0:
            reward += 0.2
        
        # Reward for finalizing
        if message.component in ["answer_drafter", "finalizer"]:
            reward += 0.5
        
        # Penalty for errors
        if message.error:
            reward -= 1.0
        
        return reward

