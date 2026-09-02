"""
Trace exporter for generating training-ready output files.

This module exports simulation results in a format compatible with
fine-tuning and RL training, following the run_exports/ structure.
"""

import json
import uuid
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime, timezone
from loguru import logger


class TraceExporter:
    """Export simulation traces in training-ready format"""
    
    def __init__(self, output_dir: str = "./data/simulation_output"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # File handles for streaming
        self._traces_file = None
        self._supervised_file = None
        self._trajectories_file = None
    
    def _ensure_stream_files_open(self):
        """Ensure streaming file handles are open"""
        if self._traces_file is None:
            self._traces_file = open(self.output_dir / "traces.jsonl", 'w')
        if self._supervised_file is None:
            self._supervised_file = open(self.output_dir / "supervised.jsonl", 'w')
        if self._trajectories_file is None:
            self._trajectories_file = open(self.output_dir / "trajectories.jsonl", 'w')
    
    def close_streams(self):
        """Close all streaming file handles"""
        if self._traces_file:
            self._traces_file.close()
            self._traces_file = None
        if self._supervised_file:
            self._supervised_file.close()
            self._supervised_file = None
        if self._trajectories_file:
            self._trajectories_file.close()
            self._trajectories_file = None
    
    def stream_trace_entry(self, trace_entry: Dict[str, Any]):
        """Stream a single trace entry to traces.jsonl"""
        self._ensure_stream_files_open()
        self._traces_file.write(json.dumps(trace_entry) + '\n')
        self._traces_file.flush()
    
    def stream_supervised_entry(self, supervised_entry: Dict[str, Any]):
        """Stream a single supervised entry to supervised.jsonl"""
        self._ensure_stream_files_open()
        self._supervised_file.write(json.dumps(supervised_entry) + '\n')
        self._supervised_file.flush()
    
    def stream_trajectory_entry(self, trajectory_entry: Dict[str, Any]):
        """Stream a single trajectory entry to trajectories.jsonl"""
        self._ensure_stream_files_open()
        self._trajectories_file.write(json.dumps(trajectory_entry) + '\n')
        self._trajectories_file.flush()
    
    def export_run(
        self,
        template: Any,
        query: str,
        expected_answer: str,
        workflow: Any,
        context: Any,
        result: Dict[str, Any],
        run_id: str = None
    ) -> str:
        """
        Export a complete simulation run with all required files.
        
        Returns:
            run_uuid: The UUID for this run
        """
        # Generate UUID for this run
        run_uuid = run_id or str(uuid.uuid4())
        run_dir = self.output_dir / run_uuid
        run_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Exporting run to {run_dir}")
        
        # Extract metrics from result and context
        metrics = self._extract_metrics(result, context)
        
        # Generate all required files
        self._write_manifest(run_dir, run_uuid, template, query, expected_answer, metrics)
        self._write_config(run_dir, run_uuid, template, query, expected_answer, workflow)
        self._write_stats(run_dir, run_uuid, template, metrics, context)
        self._write_traces(run_dir, context)
        self._write_supervised(run_dir, context)
        self._write_trajectories(run_dir, context)
        
        logger.info(f"✓ Run {run_uuid} exported successfully")
        return run_uuid
    
    def export_summary_files(
        self,
        template: Any,
        query: str,
        expected_answer: str,
        workflow: Any,
        context: Any,
        result: Dict[str, Any],
        run_id: str = None
    ):
        """
        Export summary files (manifest, config, stats) after streaming is complete.
        
        Call this AFTER all traces have been streamed during execution.
        """
        # Close streaming files
        self.close_streams()
        
        run_uuid = run_id or str(uuid.uuid4())[:8]
        
        logger.info(f"Exporting summary files for {run_uuid}")
        
        # Extract metrics from result and context
        metrics = self._extract_metrics(result, context)
        
        # Generate summary files
        self._write_manifest(self.output_dir, run_uuid, template, query, expected_answer, metrics)
        self._write_config(self.output_dir, run_uuid, template, query, expected_answer, workflow, context)
        self._write_stats(self.output_dir, run_uuid, template, metrics, context)
        
        logger.info(f"✓ Summary files for {run_uuid} exported successfully")
    
    def _extract_metrics(self, result: Dict[str, Any], context: Any) -> Dict[str, Any]:
        """Extract key metrics from execution"""
        total_latency = 0
        total_tokens = 0
        
        if hasattr(context, 'messages'):
            for msg in context.messages:
                msg_dict = msg if isinstance(msg, dict) else self._msg_to_dict(msg)
                total_latency += msg_dict.get("execution_time_ms", 0)
                total_tokens += msg_dict.get("tokens", 0)
        
        return {
            "success": result.get("success", False),
            "iterations": result.get("iterations", 0),
            "total_steps": len(context.messages) if hasattr(context, 'messages') else 0,
            "evidence_count": context.evidence_count if hasattr(context, 'evidence_count') else 0,
            "synthesis": result.get("synthesis"),
            "similarity": result.get("similarity", 0.0),
            "total_latency_ms": total_latency,
            "total_tokens": total_tokens
        }
    
    def _write_manifest(
        self, 
        run_dir: Path, 
        run_uuid: str, 
        template: Any, 
        query: str,
        expected_answer: str,
        metrics: Dict[str, Any]
    ):
        """Write manifest.json - high-level summary"""
        teacher_id = template.teacher_models[0].model_id if template.teacher_models else "unknown"
        
        # Calculate total cost (rough estimate: $0.01 per 1K tokens)
        total_cost = (metrics.get("total_tokens", 0) / 1000) * 0.01
        
        manifest = {
            "run_uuid": run_uuid,
            "teacher_id": teacher_id,
            "query": query,
            "success": metrics.get("success", False),
            "total_steps": metrics.get("total_steps", 0),
            "total_latency_ms": metrics.get("total_latency_ms", 0.0),
            "total_tokens": metrics.get("total_tokens", 0),
            "total_cost": total_cost,
            "iterations": metrics.get("iterations", 0),
            "answer_match_score": metrics.get("similarity", 0.0),
            "error_flags": [],
            "created_at": datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z'),
            "files": {
                "config": "config.json",
                "manifest": "manifest.json",
                "traces": "traces.jsonl",
                "trajectories": "trajectories.jsonl",
                "supervised": "supervised.jsonl",
                "stats": "stats.json"
            }
        }
        
        with open(run_dir / "manifest.json", 'w') as f:
            json.dump(manifest, f, indent=2)
    
    def _write_config(
        self,
        run_dir: Path,
        run_uuid: str,
        template: Any,
        query: str,
        expected_answer: str,
        workflow: Any,
        context: Any = None
    ):
        """Write config.json - run configuration"""
        teacher_id = template.teacher_models[0].model_id if template.teacher_models else "unknown"
        
        # Convert workflow components to config format
        blocks = []
        if hasattr(workflow, 'components') and workflow.components:
            for comp in workflow.components:
                blocks.append({
                    "type": comp.get("type", "unknown"),
                    "label": comp.get("type", "").replace("_", " ").title(),
                    "llm_decided": False  # Fixed workflow
                })
        
        # Extract search configuration (prefer what actually ran from context)
        search_endpoint = "chatnoir"
        search_index = "cw12"
        if hasattr(template, 'retrieval') and template.retrieval:
            search_endpoint = getattr(template.retrieval, 'type', search_endpoint)
            search_index = getattr(template.retrieval, 'corpus', search_index)
        # Override from context if available
        try:
            if context and hasattr(context, "messages") and context.messages:
                for msg in context.messages:
                    msg_dict = msg if isinstance(msg, dict) else self._msg_to_dict(msg)
                    tool = msg_dict.get("component_type") or (msg_dict.get("action", {}) or {}).get("tool") or ""
                    if tool in ["chatnoir_retriever", "opensearch_retriever", "vector_retriever"]:
                        tool_input = msg_dict.get("tool_input", {}) or {}
                        # For ChatNoir
                        corpus = tool_input.get("corpus")
                        if corpus:
                            search_endpoint = "chatnoir"
                            search_index = corpus
                            break
        except Exception:
            pass
        
        config = {
            "run_uuid": run_uuid,
            "run_id": f"{teacher_id.replace('-', '_')}_{int(datetime.now().timestamp())}",
            "teacher_id": teacher_id,
            "query": query,
            "expected_answer": expected_answer,
            "workflow_config": {
                "blocks": blocks,
                "answer_match_threshold": template.similarity_threshold,
                "max_iterations": template.max_iterations
            },
            "search_endpoint": search_endpoint,
            "search_index": search_index,
            "created_at": datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')
        }
        
        with open(run_dir / "config.json", 'w') as f:
            json.dump(config, f, indent=2)
    
    def _write_stats(
        self,
        run_dir: Path,
        run_uuid: str,
        template: Any,
        metrics: Dict[str, Any],
        context: Any
    ):
        """Write stats.json - detailed statistics"""
        teacher_id = template.teacher_models[0].model_id if template.teacher_models else "unknown"
        
        # Aggregate tool usage, latency, and tokens
        tool_usage = {}
        latency_by_tool = {}
        tokens_by_tool = {}
        evidence_progression = []
        
        if hasattr(context, 'messages'):
            for msg in context.messages:
                # Use unified mapping
                msg_dict = msg if isinstance(msg, dict) else self._msg_to_dict(msg)
                # Prefer component_type, else fall back to action.tool if present
                tool = msg_dict.get("component_type") or (msg_dict.get("action", {}) or {}).get("tool") or "unknown"
                
                # Count usage
                tool_usage[tool] = tool_usage.get(tool, 0) + 1
                
                # Track latency
                latency_ms = msg_dict.get("execution_time_ms", 0)
                if tool not in latency_by_tool:
                    latency_by_tool[tool] = {
                        "count": 0,
                        "total_ms": 0.0,
                        "min_ms": float('inf'),
                        "max_ms": 0.0
                    }
                
                latency_by_tool[tool]["count"] += 1
                latency_by_tool[tool]["total_ms"] += latency_ms
                latency_by_tool[tool]["min_ms"] = min(latency_by_tool[tool]["min_ms"], latency_ms)
                latency_by_tool[tool]["max_ms"] = max(latency_by_tool[tool]["max_ms"], latency_ms)
                
                # Track tokens
                tokens = msg_dict.get("tokens", 0)
                if tool not in tokens_by_tool:
                    tokens_by_tool[tool] = {"count": 0, "total": 0}
                
                tokens_by_tool[tool]["count"] += 1
                tokens_by_tool[tool]["total"] += tokens
                
                # Evidence progression
                evidence_progression.append({
                    "step_id": msg_dict.get("turn", 0),
                    "evidence_count": msg_dict.get("evidence_count", 0),
                    "evidence_available_pct": 0.0,
                    "retrieval_overlap": 0.0
                })
        
        # Calculate averages
        for tool, stats in latency_by_tool.items():
            stats["avg_ms"] = stats["total_ms"] / stats["count"] if stats["count"] > 0 else 0.0
        
        for tool, stats in tokens_by_tool.items():
            stats["avg"] = stats["total"] / stats["count"] if stats["count"] > 0 else 0.0
        
        stats = {
            "teacher_id": teacher_id,
            "run_uuid": run_uuid,
            "success": metrics.get("success", False),
            "answer_match_score": metrics.get("similarity", 0.0),
            "total_steps": metrics.get("total_steps", 0),
            "iterations": metrics.get("iterations", 0),
            "error_flags": [],
            "total_latency_ms": metrics.get("total_latency_ms", 0.0),
            "total_tokens": metrics.get("total_tokens", 0),
            "total_cost": (metrics.get("total_tokens", 0) / 1000) * 0.01,
            "avg_latency_per_step_ms": (
                metrics.get("total_latency_ms", 0.0) / max(metrics.get("total_steps", 0), 1)
            ),
            "avg_tokens_per_step": (
                metrics.get("total_tokens", 0) / max(metrics.get("total_steps", 0), 1)
            ),
            "tool_usage": tool_usage,
            "rationale_tags": {},  # TODO: Extract from messages
            "decision_labels": {},  # TODO: Extract from messages
            "query_evolution": [],  # TODO: Track query changes
            "evidence_progression": evidence_progression,
            "latency_by_tool": latency_by_tool,
            "tokens_by_tool": tokens_by_tool,
            "final_evidence_count": metrics.get("evidence_count", 0),
            "avg_evidence_overlap": 0.0,  # TODO: Calculate
            "created_at": datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')
        }
        
        with open(run_dir / "stats.json", 'w') as f:
            json.dump(stats, f, indent=2)
    
    def _write_traces(self, run_dir: Path, context: Any):
        """Write traces.jsonl - full step-by-step execution trace"""
        with open(run_dir / "traces.jsonl", 'w') as f:
            if hasattr(context, 'messages'):
                for msg in context.messages:
                    # Convert Message to dict
                    msg_dict = msg if isinstance(msg, dict) else self._msg_to_dict(msg)
                    trace_entry = {
                        "step_id": msg_dict.get("turn", 0),
                        "goal": msg_dict.get("goal", ""),
                        "action": {
                            "tool": msg_dict.get("component_type", "unknown"),
                            "parameters": msg_dict.get("parameters", {})
                        },
                        "rationale_tag": msg_dict.get("rationale_tag", ""),
                        "operator_intent": msg_dict.get("operator_intent", ""),
                        "stop_condition": msg_dict.get("stop_condition", "CONTINUE"),
                        "timestamp": msg_dict.get("timestamp", datetime.now().timestamp()),
                        "private_reasoning": msg_dict.get("private_reasoning", ""),
                        "llm_input": msg_dict.get("llm_input", None),
                        "llm_output": msg_dict.get("llm_output", None),
                        "tool_input": msg_dict.get("tool_input", {}),
                        "tool_output": msg_dict.get("tool_output", {}),
                        "evidence_retrieved": msg_dict.get("evidence_retrieved", None),
                        "evidence_count": msg_dict.get("evidence_count", 0),
                        "execution_time_ms": msg_dict.get("execution_time_ms", 0.0),
                        "error": msg_dict.get("error", None)
                    }
                    f.write(json.dumps(trace_entry) + '\n')
    
    def _write_supervised(self, run_dir: Path, context: Any):
        """Write supervised.jsonl - LLM input/output pairs for fine-tuning"""
        with open(run_dir / "supervised.jsonl", 'w') as f:
            if hasattr(context, 'messages'):
                for msg in context.messages:
                    msg_dict = msg if isinstance(msg, dict) else self._msg_to_dict(msg)
                    if msg_dict.get("llm_input") and msg_dict.get("llm_output"):
                        supervised_entry = {
                            "step_id": msg_dict.get("turn", 0),
                            "input": msg_dict.get("llm_input", ""),
                            "output": msg_dict.get("llm_output", ""),
                            "tool": msg_dict.get("component_type", "unknown"),
                            "rationale_tag": msg_dict.get("rationale_tag", ""),
                            "decision_label": msg_dict.get("stop_condition", "CONTINUE"),
                            "latency_ms": msg_dict.get("execution_time_ms", 0.0),
                            "tokens": msg_dict.get("tokens", 0)
                        }
                        f.write(json.dumps(supervised_entry) + '\n')
    
    def _write_trajectories(self, run_dir: Path, context: Any):
        """Write trajectories.jsonl - state-action-reward for RL training"""
        with open(run_dir / "trajectories.jsonl", 'w') as f:
            if hasattr(context, 'messages'):
                for i, msg in enumerate(context.messages):
                    msg_dict = msg if isinstance(msg, dict) else self._msg_to_dict(msg)
                    # Get next state (if exists)
                    next_msg = context.messages[i + 1] if i + 1 < len(context.messages) else None
                    next_msg_dict = next_msg if isinstance(next_msg, dict) else self._msg_to_dict(next_msg) if next_msg else None
                    
                    trajectory_entry = {
                        "state": {
                            "step_id": msg_dict.get("turn", 0),
                            "query": context.query if hasattr(context, 'query') else "",
                            "evidence_count": msg_dict.get("evidence_count", 0),
                            "evidence_available_pct": 0.0,
                            "state_stage": self._determine_stage(msg_dict),
                            "previous_tools": self._get_previous_tools(context.messages, i)
                        },
                        "action": {
                            "tool": msg_dict.get("component_type", "unknown"),
                            "parameters": msg_dict.get("parameters", {}),
                            "rationale_tag": msg_dict.get("rationale_tag", "")
                        },
                        "reward": self._calculate_reward(msg_dict, next_msg_dict),
                        "next_state": {
                            "step_id": next_msg_dict.get("turn", 0) if next_msg_dict else 0,
                            "query": context.query if hasattr(context, 'query') else "",
                            "evidence_count": next_msg_dict.get("evidence_count", 0) if next_msg_dict else 0,
                            "evidence_available_pct": 0.0,
                            "state_stage": self._determine_stage(next_msg_dict) if next_msg_dict else "done"
                        },
                        "done": next_msg is None or msg_dict.get("stop_condition") == "FINISH"
                    }
                    f.write(json.dumps(trajectory_entry) + '\n')
    
    def _determine_stage(self, msg: Dict[str, Any]) -> str:
        """Determine workflow stage from message"""
        tool = msg.get("component_type", "")
        
        if tool in ["planner", "query_formulator"]:
            return "initial"
        elif tool in ["opensearch_retriever", "chatnoir_retriever", "vector_retriever"]:
            return "retrieving"
        elif tool in ["reranker", "filter", "deduplicator", "extractor"]:
            return "processing"
        elif tool in ["answer_drafter", "finalizer"]:
            return "synthesizing"
        elif tool in ["fact_checker", "attribution_gate"]:
            return "verifying"
        else:
            return "unknown"
    
    def _get_previous_tools(self, messages: List, current_idx: int) -> List[str]:
        """Get list of previously used tools"""
        previous = []
        for msg in messages[:current_idx]:
            msg_dict = msg if isinstance(msg, dict) else self._msg_to_dict(msg)
            previous.append(msg_dict.get("component_type", "unknown"))
        return previous
    
    def _msg_to_dict(self, msg: Any) -> Dict[str, Any]:
        """Convert Message object to dictionary"""
        if msg is None:
            return {}
        return {
            "turn": getattr(msg, 'turn', 0),
            "component": getattr(msg, 'component', ''),
            "component_type": getattr(msg, 'component', ''),  # Alias
            "thought": getattr(msg, 'thought', ''),
            "action": getattr(msg, 'action', {}),
            "observation": getattr(msg, 'observation', ''),
            "citations": getattr(msg, 'citations', []),
            "parameters": getattr(msg, 'action', {}).get('parameters', {}) if hasattr(msg, 'action') else {},
            "execution_time_ms": getattr(msg, 'execution_time_ms', 0),
            "tokens": getattr(msg, 'tokens', 0),
            "evidence_count": getattr(msg, 'evidence_count', 0),
            "rationale_tag": getattr(msg, 'rationale_tag', ''),
            "stop_condition": getattr(msg, 'stop_condition', 'CONTINUE'),
            "llm_input": getattr(msg, 'llm_input', None),
            "llm_output": getattr(msg, 'llm_output', None),
            "tool_input": getattr(msg, 'tool_input', {}),
            "tool_output": getattr(msg, 'tool_output', {}),
            "error": getattr(msg, 'error', None)
        }
    
    def _calculate_reward(
        self, 
        current_msg: Dict[str, Any], 
        next_msg: Dict[str, Any] = None
    ) -> float:
        """Calculate reward for this action (simplified)"""
        # Positive reward for successful actions
        if current_msg.get("success", True) and not current_msg.get("error"):
            reward = 1.0
        else:
            reward = -0.5
        
        # Bonus for retrieving evidence
        if current_msg.get("evidence_count", 0) > 0:
            reward += 0.2
        
        # Bonus for finishing successfully
        if current_msg.get("stop_condition") == "FINISH":
            reward += 2.0
        
        return reward

