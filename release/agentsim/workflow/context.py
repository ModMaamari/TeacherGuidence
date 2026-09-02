"""
Workflow context (formerly Blackboard) for managing shared state.

The context is passed through all components in a workflow
and maintains the shared state including evidence, queries, and metadata.
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from loguru import logger


@dataclass
class EvidenceSpan:
    """
    A single piece of evidence retrieved during the workflow.
    
    Attributes:
        source: Where this evidence came from (opensearch, vector, chatnoir)
        id: Unique identifier for this span
        text: The actual text content
        start: Start position in original document
        end: End position in original document
        doc_meta: Additional metadata (title, score, etc.)
    """
    source: str
    id: str
    text: str
    start: int = 0
    end: int = 0
    doc_meta: Dict[str, Any] = field(default_factory=dict)
    
    def __hash__(self):
        return hash(self.id)
    
    def __eq__(self, other):
        if not isinstance(other, EvidenceSpan):
            return False
        return self.id == other.id


@dataclass
class Message:
    """
    A single message in the workflow execution trace.
    
    Attributes:
        turn: Turn number in the workflow
        component: Name of the component that generated this message
        thought: Internal reasoning/explanation
        action: Action taken (tool and parameters)
        observation: Result of the action
        citations: Evidence spans referenced
        verdict: Execution verdict (PROCEED, REPLAN, FAIL, FINISH)
        latency_ms: Time taken for this step (alias for execution_time_ms)
        rationale_tag: Reasoning tag (e.g., BREADTH, DEPTH, RERANK)
        stop_condition: Stopping condition (CONTINUE, FINISH, ERROR)
        llm_input: Raw LLM input (prompt)
        llm_output: Raw LLM output (completion)
        tool_input: Tool-specific input parameters
        tool_output: Tool-specific output data
        execution_time_ms: Execution time in milliseconds
        tokens: Token count for LLM calls
        evidence_count: Number of evidence items retrieved/used
        error: Error message if any
    """
    turn: int
    component: str
    thought: str = ""
    action: Dict[str, Any] = field(default_factory=dict)
    observation: Dict[str, Any] = field(default_factory=dict)
    citations: List[EvidenceSpan] = field(default_factory=list)
    verdict: str = "PROCEED"
    latency_ms: float = 0.0
    
    # Additional fields for trace export
    rationale_tag: str = ""
    stop_condition: str = "CONTINUE"
    llm_input: Optional[str] = None
    llm_output: Optional[str] = None
    tool_input: Dict[str, Any] = field(default_factory=dict)
    tool_output: Dict[str, Any] = field(default_factory=dict)
    execution_time_ms: float = 0.0
    tokens: int = 0
    evidence_count: int = 0
    error: Optional[str] = None


class WorkflowContext:
    """
    Shared context that flows through all components in a workflow.
    
    This is the central data structure that maintains:
    - Input query and task metadata
    - Evidence collected from retrieval
    - Intermediate results and messages
    - Final outputs
    
    Components read from and write to this context.
    """
    
    def __init__(
        self,
        task_id: str,
        query: str,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize workflow context.
        
        Args:
            task_id: Unique identifier for this task
            query: User query to answer
            metadata: Additional metadata
        """
        self.task_id = task_id
        self.query = query
        self.metadata = metadata or {}
        
        # Evidence storage
        self.evidence_store: List[EvidenceSpan] = []
        self._evidence_by_id: Dict[str, EvidenceSpan] = {}
        
        # Query tracking
        self.queries: List[str] = []
        self.subgoals: List[str] = []
        
        # Message history
        self.messages: List[Message] = []
        
        # Step counter
        self._step_count = 0
    
    def add_evidence(self, span: EvidenceSpan) -> None:
        """
        Add an evidence span to the store.
        
        Args:
            span: Evidence span to add
        """
        # Deduplicate by ID
        if span.id not in self._evidence_by_id:
            self.evidence_store.append(span)
            self._evidence_by_id[span.id] = span
            logger.debug(f"Added evidence: {span.id} ({len(span.text)} chars)")
    
    def get_evidence_by_id(self, span_id: str) -> Optional[EvidenceSpan]:
        """
        Retrieve an evidence span by ID.
        
        Args:
            span_id: Evidence span identifier
            
        Returns:
            EvidenceSpan or None if not found
        """
        return self._evidence_by_id.get(span_id)
    
    def add_message(self, message: Message) -> None:
        """
        Add a message to the execution trace.
        
        Args:
            message: Message to add
        """
        self.messages.append(message)
        logger.debug(f"Added message: turn={message.turn}, component={message.component}")
    
    def should_replan(self) -> bool:
        """
        Check if the workflow should replan based on recent verdicts.
        
        Returns:
            True if replanning is needed
        """
        if not self.messages:
            return False
        
        # Check last 3 messages for failures
        recent_messages = self.messages[-3:]
        failures = sum(1 for msg in recent_messages if msg.verdict in ["FAIL", "REPLAN"])
        
        return failures >= 2
    
    def get_step_count(self) -> int:
        """Get the current step count."""
        return self._step_count
    
    def increment_step(self) -> int:
        """Increment and return the step count."""
        self._step_count += 1
        return self._step_count
    
    def get_state_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the current context state.
        
        Returns:
            Dictionary with state summary
        """
        return {
            "task_id": self.task_id,
            "query": self.query,
            "evidence_count": len(self.evidence_store),
            "query_count": len(self.queries),
            "subgoal_count": len(self.subgoals),
            "message_count": len(self.messages),
            "step_count": self._step_count,
            "metadata": self.metadata
        }
    
    def clear_evidence(self) -> None:
        """Clear all evidence from the store."""
        self.evidence_store.clear()
        self._evidence_by_id.clear()
        logger.debug("Cleared evidence store")
    
    def __repr__(self) -> str:
        return (f"WorkflowContext(task_id={self.task_id}, "
                f"evidence={len(self.evidence_store)}, "
                f"messages={len(self.messages)})")

