"""Standard Mode: Fixed workflow with iterative execution"""

from typing import Dict, Any, Optional
from loguru import logger
from math import sqrt

from agentsim.workflow.executor import WorkflowExecutor
from agentsim.config import config
from agentsim.simulation.verifier import RealtimeVerifier
from agentsim.simulation.schema import ModelConfig
from agentsim.teacher_guidance.provenance import config_hash


class StandardRunner:
    """Standard iterative execution with fixed workflow"""
    
    def __init__(self, template, clients, workflow_executor=None):
        self.template = template
        self.clients = clients
        self.workflow_executor = workflow_executor
        
        # Consultant configuration (per component type)
        self.consultant_lookup = {model.name: model for model in template.consultant_models or []}
        self.parallel_consultants = {}
        for parallel_cfg in template.parallel_execution or []:
            consultants = [
                self.consultant_lookup[name]
                for name in parallel_cfg.consultant_models
                if name in self.consultant_lookup
            ]
            if consultants:
                self.parallel_consultants[parallel_cfg.component_type] = consultants
        
        # Verification
        self.verifier = None
        self.verifier_model: ModelConfig | None = None
        if template.verification and template.verification.enabled:
            if template.verifier_model:
                self.verifier_model = template.verifier_model
            else:
                self.verifier_model = ModelConfig(
                    name="verifier",
                    model_id=config.VERIFIER_MODEL,
                    role="verifier",
                    temperature=config.VERIFIER_TEMPERATURE,
                )
            verifier_llm = self.clients.get("llm_client")
            if verifier_llm and self.verifier_model:
                self.verifier = RealtimeVerifier(
                    verifier_llm,
                    template.verification,
                    self.verifier_model.model_id,
                )
        
    async def run(
        self,
        workflow,
        dataset_sample: Dict[str, Any],
        initial_metadata: Dict[str, Any] = None,
        sample_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Execute mode 1 simulation
        
        Args:
            workflow: Workflow to execute
            dataset_sample: Sample with query and answer
            initial_metadata: Optional metadata to inject into context (e.g., knowledge base)
        """
        
        query = dataset_sample["query"]
        gold_answer = dataset_sample["answer"]

        # Inject Teacher Guidance metadata when the template defines mode_config.
        # This is additive: workflows that ignore these keys are unaffected.
        mode_config = self.template.mode_config or {}
        if mode_config:
            initial_metadata = dict(initial_metadata or {})
            initial_metadata.setdefault("sample_id", sample_id or dataset_sample.get("id"))
            initial_metadata.update({
                "dataset_sample": dataset_sample,
                # Dataset provenance: prefer the question row's own `source`/`split` (every
                # converter stamps them) and let mode_config override. Without this the
                # exporter cannot tell HotpotQA traces from MuSiQue ones.
                "dataset": (mode_config.get("dataset")
                            or dataset_sample.get("source")
                            or "hotpotqa"),
                "dataset_split": (mode_config.get("dataset_split")
                                  or dataset_sample.get("split")
                                  or ""),
                # Sentence- vs paragraph-level gold differs by source and decides whether
                # supporting_fact_recall is meaningful for this episode.
                "gold_granularity": dataset_sample.get("gold_granularity", "sentence"),
                "gold": dataset_sample.get("gold", {"answer": dataset_sample.get("answer")}),
                "gold_answer": dataset_sample.get("answer"),
                "retrieval_scope": dataset_sample.get("retrieval_scope", {}),
                "budget": mode_config.get("budget"),
                "student_model": mode_config.get("student_model"),
                "teacher_model": mode_config.get("teacher_model"),
                "guidance": mode_config.get("guidance", {"level": 0}),
                "plan_review_config": mode_config.get("plan_review", {}),
                "corpus_path": mode_config.get("corpus_path"),
                "retrieval_backend": mode_config.get("retrieval_backend", "hotpot_local"),
                "skip_teacher": mode_config.get("skip_teacher", False),
                # Identifies the exact generation configuration this episode came from.
                "config_hash": config_hash(mode_config),
            })
            # Optional per-call tuning (only injected when explicitly set, so existing
            # templates keep the component defaults). A reasoning-model teacher such as
            # a reasoning model needs a larger teacher_max_tokens because its
            # chain-of-thought counts against the completion budget before the JSON.
            for _k in (
                "teacher_max_tokens", "teacher_max_tokens_retry",
                "student_max_tokens", "student_max_repair_attempts",
                "teacher_max_repair_attempts", "teacher_temperature", "student_temperature",
                "teacher_router", "student_use_response_schema", "disclose_budget",
                "wiki_enabled", "wiki_mode",
            ):
                if _k in mode_config:
                    initial_metadata[_k] = mode_config[_k]

        # Use the workflow executor provided (with trace exporter) or create new one
        if self.workflow_executor:
            self.workflow_executor.parallel_consultants = self.parallel_consultants
            executor = self.workflow_executor
        else:
            executor = WorkflowExecutor(
                **self.clients,
                parallel_consultants=self.parallel_consultants
            )
        
        # Iterative execution
        for iteration in range(1, self.template.max_iterations + 1):
            logger.info(f"Iteration {iteration}/{self.template.max_iterations}")
            
            # Pass initial_metadata to executor so it's available in context
            context = await executor.execute(workflow, query, metadata=initial_metadata)
            
            # Inject consultant models into context for planning components
            if self.template.consultant_models:
                context.consultant_models = self.template.consultant_models
            
            # Get synthesis
            synthesis = context.metadata.get("final_answer") or context.metadata.get("draft_answer")
            
            if not synthesis:
                logger.warning(f"No synthesis generated in iteration {iteration}")
                continue
            
            # Real-time verification (if enabled)
            if self.verifier and getattr(context, "evidence_store", None):
                verification_result = await self.verifier.verify_synthesis(
                    synthesis,
                    context.evidence_store,
                    context
                )
                context.metadata["verification_result"] = verification_result
                
                if (
                    verification_result.get("flagged")
                    and self.template.verification.flag_hallucinations
                ):
                    logger.warning("Verification flagged synthesis as hallucinated; retrying iteration.")
                    continue
                else:
                    logger.info("Verification passed without flags.")
            
            # Compare with gold answer
            similarity = await self._compute_similarity(synthesis, gold_answer)
            
            logger.info(f"Similarity: {similarity:.3f} (threshold: {self.template.similarity_threshold})")
            
            if similarity >= self.template.similarity_threshold:
                logger.info(f"✓ Threshold met after {iteration} iterations")
                return {
                    "success": True,
                    "iterations": iteration,
                    "synthesis": synthesis,
                    "similarity": similarity,
                    "context": context
                }
        
        # Max iterations reached
        return {
            "success": False,
            "iterations": self.template.max_iterations,
            "synthesis": synthesis if 'synthesis' in locals() else None,
            "similarity": similarity if 'similarity' in locals() else 0.0,
            "context": context if 'context' in locals() else None
        }
    
    async def _compute_similarity(self, synthesis: str, gold: str) -> float:
        """Compute similarity based on configured metric"""
        
        # Handle empty strings
        if not synthesis or not synthesis.strip():
            logger.warning("Synthesis is empty, returning similarity 0.0")
            return 0.0
        
        if not gold or not gold.strip():
            logger.warning("Gold answer is empty, returning similarity 1.0 (no comparison needed)")
            return 1.0  # No gold answer to compare against, so pass
        
        metric = self.template.similarity_metric
        
        if metric.value == "embedding_cosine":
            try:
                return await self._embedding_similarity(synthesis, gold)
            except Exception as e:
                logger.error(f"Embedding similarity failed: {e}")
                # Fallback to token overlap
                return self._token_overlap(synthesis, gold)
        elif metric.value == "token_overlap":
            return self._token_overlap(synthesis, gold)
        else:
            return 0.5
    
    async def _embedding_similarity(self, text1: str, text2: str) -> float:
        """Cosine similarity of embeddings"""
        
        # Double-check for empty strings
        if not text1 or not text1.strip() or not text2 or not text2.strip():
            logger.warning("Empty text provided to embedding similarity")
            return 0.0
        
        llm = self.clients.get("llm_client")
        if not llm:
            raise ValueError("LLM client not available for embeddings")
        
        emb1 = await llm.get_embedding(text1)
        emb2 = await llm.get_embedding(text2)
        
        if not emb1 or not emb2:
            raise ValueError("Empty embeddings returned")
        
        dot = sum(a * b for a, b in zip(emb1, emb2))
        norm1 = sqrt(sum(a * a for a in emb1))
        norm2 = sqrt(sum(b * b for b in emb2))
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return dot / (norm1 * norm2)
    
    def _token_overlap(self, text1: str, text2: str) -> float:
        """Token overlap similarity"""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        overlap = len(words1 & words2)
        return overlap / len(words1 | words2)

