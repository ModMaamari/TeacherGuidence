"""Exploratory Mode: Knowledge expansion through iterative exploration"""

import json
import random
from pathlib import Path
from typing import Dict, Any, List, Set, Optional
from loguru import logger
from datetime import datetime


class KnowledgeBase:
    """Growing knowledge base for exploratory mode"""
    
    def __init__(self):
        self.entries: List[Dict[str, Any]] = []
        self.doc_index: Dict[str, Set[str]] = {}  # doc_id -> queries that found it
        self.query_index: Dict[str, Dict] = {}  # query -> result
    
    def add_entry(self, query: str, answer: str, evidence: List[Any], doc_ids: List[str], seed_id: Optional[str] = None):
        """Add new knowledge entry"""
        
        # Ensure answer is never None
        if answer is None:
            answer = ""
        
        # Convert evidence to JSON-serializable format
        evidence_serializable = []
        if evidence:
            for ev in evidence:
                if hasattr(ev, '__dict__'):
                    # EvidenceSpan object - convert to dict
                    evidence_serializable.append({
                        "id": getattr(ev, 'id', ''),
                        "text": getattr(ev, 'text', '')[:500],  # Truncate for storage
                        "source": getattr(ev, 'source', ''),
                        "score": getattr(ev, 'score', 0.0)
                    })
                elif isinstance(ev, dict):
                    # Already a dict
                    evidence_serializable.append({
                        "id": ev.get('id', ''),
                        "text": str(ev.get('text', ''))[:500],
                        "source": ev.get('source', ''),
                        "score": ev.get('score', 0.0)
                    })
        
        entry = {
            "query": query,
            "answer": answer,
            "evidence": evidence_serializable,
            "doc_ids": doc_ids,
            "timestamp": datetime.now().isoformat(),
            "seed_id": seed_id
        }
        self.entries.append(entry)
        self.query_index[query] = entry
        
        # Update doc index
        for doc_id in doc_ids:
            if doc_id not in self.doc_index:
                self.doc_index[doc_id] = set()
            self.doc_index[doc_id].add(query)
    
    def search(self, query: str) -> Optional[Dict]:
        """Search knowledge base for similar queries"""
        return self.query_index.get(query)
    
    def get_context(self, max_entries: int = 5) -> str:
        """Get recent knowledge as context"""
        recent = self.entries[-max_entries:] if len(self.entries) > max_entries else self.entries
        context_parts = []
        for entry in recent:
            answer = entry.get('answer') or ''
            answer_text = answer[:200] if answer else 'No answer available'
            context_parts.append(f"Q: {entry['query']}\nA: {answer_text}")
        return "\n\n".join(context_parts)
    
    def get_all_doc_ids(self) -> List[str]:
        """Get all retrieved document IDs"""
        return list(self.doc_index.keys())
    
    def get_stats(self) -> Dict[str, Any]:
        """Get knowledge base statistics"""
        return {
            "total_queries": len(self.entries),
            "unique_documents": len(self.doc_index),
            "avg_docs_per_query": sum(len(e["doc_ids"]) for e in self.entries) / max(len(self.entries), 1)
        }


class ExploratoryRunner:
    """Exploratory mode: Iterative knowledge expansion with reflection"""
    
    def __init__(self, template, clients, workflow_executor, run_uuid=None):
        self.template = template
        self.clients = clients
        self.workflow_executor = workflow_executor
        self.knowledge_base = KnowledgeBase()
        self.explored_queries: Set[str] = set()
        self.run_uuid = run_uuid  # Store run UUID for output directory
        self.current_seed_id: Optional[str] = None
        
        # Mode config
        config = template.mode_config
        self.max_explorations = config.get("max_explorations", 10)
        self.use_knowledge_base = config.get("use_knowledge_base", True)
        self.execution_mode = config.get("execution_mode", "standard")
        self.reflection_temperature = config.get("reflection_temperature", 0.8)
        self.questions_per_reflection = config.get("questions_per_reflection", 3)
        self.external_knowledge_component = config.get("external_knowledge_component")
        self.enable_consultant_reflection = config.get("enable_consultant_reflection", True)
    
    async def run(self, workflow, dataset_sample: Dict[str, Any], sample_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Execute exploratory simulation.
        
        Starts with the provided sample and explores related questions iteratively.
        """
        
        logger.info("Starting exploratory mode")
        logger.info(f"Max explorations: {self.max_explorations}")
        logger.info(f"Use knowledge base: {self.use_knowledge_base}")
        logger.info(f"Execution mode: {self.execution_mode}")
        if self.current_seed_id:
            logger.info(f"Seed ID: {self.current_seed_id}")
        
        # Use the provided sample as starting point
        initial_query = dataset_sample.get("query") or dataset_sample.get("question")
        gold_answer = dataset_sample.get("answer") or dataset_sample.get("gold_answer")
        self.current_seed_id = sample_id or dataset_sample.get("seed_id") or dataset_sample.get("uid")
        
        logger.info(f"Initial query: {initial_query}")
        
        exploration_log = []
        all_contexts = []  # Collect all contexts for aggregated stats
        teacher_model = self.template.teacher_models[0]  # Primary teacher
        
        # Queue of queries to explore
        query_queue = [initial_query]
        exploration_count = 0
        
        while exploration_count < self.max_explorations and query_queue:
            # Get next query from queue
            current_query = query_queue.pop(0)
            
            logger.info(f"\n{'='*60}")
            logger.info(f"Exploration {exploration_count+1}/{self.max_explorations}")
            logger.info(f"Query: {current_query}")
            logger.info(f"{'='*60}\n")
            
            # Check if already explored
            if current_query in self.explored_queries:
                logger.warning(f"Query already explored, skipping")
                continue
            
            self.explored_queries.add(current_query)
            exploration_count += 1
            
            # Execute workflow
            execution_result = await self._execute_query(
                workflow, 
                current_query, 
                teacher_model,
                use_kb=self.use_knowledge_base
            )
            
            # Collect context for aggregated stats
            if execution_result.get("context"):
                all_contexts.append(execution_result["context"])
            
            # Verify against external knowledge if configured
            external_verification = None
            external_answer = ""
            if self.external_knowledge_component:
                external_verification = await self._verify_with_external_knowledge(
                    current_query,
                    execution_result.get("synthesis", ""),
                    execution_result.get("context")
                )
                # Use external verification answer for similarity comparison
                external_answer = external_verification.get("external_answer", "") if external_verification else ""
            
            # Compute similarity with external verification (if available) for exploratory mode
            if external_answer and execution_result.get("synthesis"):
                similarity = await self._compute_similarity_local(
                    execution_result.get("synthesis"),
                    external_answer
                )
                execution_result["similarity_score"] = similarity
                logger.info(f"Similarity with external knowledge: {similarity:.3f}")
            
            # Get consultant reflections if enabled
            consultant_questions = []
            consultant_details = []
            if self.enable_consultant_reflection and self.template.consultant_models:
                consultant_result = await self._get_consultant_reflections(
                    current_query,
                    execution_result
                )
                if isinstance(consultant_result, dict):
                    consultant_questions = consultant_result.get("questions", [])
                    consultant_details = consultant_result.get("details", [])
                else:
                    # Backward compatibility
                    consultant_questions = consultant_result
            
            # Teacher reflection and question generation
            teacher_questions = await self._reflect_and_expand(
                current_query,
                execution_result,
                teacher_model
            )
            
            # Record exploration
            exploration_entry = {
                "iteration": exploration_count,
                "query": current_query,
                "answer": execution_result.get("synthesis", ""),
                "similarity_score": execution_result.get("similarity_score", 0.0),
                "iterations_taken": execution_result.get("iterations", 0),
                "evidence_count": len(execution_result.get("evidence", [])),
                "doc_ids": execution_result.get("doc_ids", []),
                "execution_mode_used": execution_result.get("mode_used", self.execution_mode),
                "teacher_generated_questions": teacher_questions,
                "consultant_generated_questions": consultant_questions,
                "consultant_details": consultant_details if consultant_details else None,
                "external_verification": external_verification,
                "timestamp": datetime.now().isoformat(),
                "seed_id": self.current_seed_id
            }
            exploration_log.append(exploration_entry)
            
            # Update knowledge base
            evidence = execution_result.get("evidence", [])
            doc_ids = execution_result.get("doc_ids", [])
            
            logger.debug(f"Evidence count: {len(evidence)}, Doc IDs count: {len(doc_ids)}")
            
            self.knowledge_base.add_entry(
                query=current_query,
                answer=execution_result.get("synthesis", ""),
                evidence=evidence,
                doc_ids=doc_ids,
                seed_id=self.current_seed_id
            )
            
            logger.info(f"Generated {len(teacher_questions)} teacher questions")
            logger.info(f"Generated {len(consultant_questions)} consultant questions")
            
            # Combine all new questions
            all_new_questions = teacher_questions + consultant_questions
            
            # Filter already explored and add to queue
            unexplored = [q for q in all_new_questions if q not in self.explored_queries]
            
            if unexplored:
                # Add unexplored questions to the queue
                query_queue.extend(unexplored)
                logger.info(f"Added {len(unexplored)} new questions to exploration queue (queue size: {len(query_queue)})")
            else:
                logger.info("No new questions generated")
        
        # Save exploration artifacts in the run directory (alongside dataset folders)
        if self.run_uuid:
            # Save directly in the run_uuid directory
            output_path = Path(self.template.output_dir) / self.run_uuid
        else:
            # Fallback: create a separate directory (shouldn't happen in normal use)
            output_path = Path(self.template.output_dir) / f"exploratory_{self.template.id}"
        
        # Ensure directory exists
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save exploration log
        self._save_exploration_log(output_path, exploration_log)
        
        # Save knowledge graph
        self._save_knowledge_graph(output_path)
        # Save coverage metrics
        self._save_coverage_metrics(output_path, exploration_log)
        
        logger.info(f"Saved exploration artifacts to: {output_path}")
        
        # Summary
        kb_stats = self.knowledge_base.get_stats()
        
        # Aggregate stats from all explorations for trace exporter
        total_iterations = sum(e.get("iterations_taken", 0) for e in exploration_log)
        total_evidence = sum(e.get("evidence_count", 0) for e in exploration_log)
        avg_similarity = sum(e.get("similarity_score", 0) for e in exploration_log) / max(len(exploration_log), 1)
        
        # Create aggregated context combining all exploration contexts
        aggregated_context = None
        if all_contexts:
            # Create a synthetic context that aggregates all messages from all explorations
            from agentsim.workflow.context import WorkflowContext
            import time
            aggregated_context = WorkflowContext(
                task_id=f"exploratory_{int(time.time())}",
                query=initial_query,
                metadata={
                    "mode": "exploratory",
                    "seed_id": self.current_seed_id,
                    "explorations_completed": len(exploration_log),
                    "total_iterations": total_iterations,
                    "total_evidence": total_evidence,
                    "avg_similarity": avg_similarity,
                    "knowledge_base_stats": kb_stats
                }
            )
            
            # Aggregate all messages from all contexts
            for ctx in all_contexts:
                if hasattr(ctx, 'messages'):
                    for msg in ctx.messages:
                        aggregated_context.messages.append(msg)
            
            logger.info(f"Aggregated {len(aggregated_context.messages)} messages from {len(all_contexts)} explorations")
        
        return {
            "mode": "exploratory",
            "seed_id": self.current_seed_id,
            "success": len(exploration_log) > 0,  # Success if we completed at least one exploration
            "explorations_completed": len(exploration_log),
            "iterations": total_iterations,  # For trace exporter
            "queries_explored": list(self.explored_queries),
            "unique_documents_retrieved": len(self.knowledge_base.get_all_doc_ids()),
            "knowledge_base_stats": kb_stats,
            "exploration_log": exploration_log,
            "output_dir": str(output_path),
            "synthesis": exploration_log[0].get("answer", "") if exploration_log else "",  # Use first exploration answer
            "similarity": avg_similarity,  # For trace exporter
            "context": aggregated_context  # Aggregated context with all messages for trace exporter
        }
    
    def _save_coverage_metrics(self, output_path: Path, exploration_log: List[Dict]):
        """Compute and save exploration coverage and overlap metrics with aggregation support."""
        seed_id = self.current_seed_id or "seed"
        cov_file = output_path / "coverage.json"
        try:
            if cov_file.exists():
                try:
                    with open(cov_file, "r") as f:
                        coverage = json.load(f)
                except Exception:
                    coverage = {}
            else:
                coverage = {}
            
            by_seed = coverage.get("by_seed", {})
            
            # Remove existing data for this seed to avoid duplicates
            if seed_id in by_seed:
                by_seed.pop(seed_id, None)
            
            seed_per_iter = []
            cumulative_docs: Set[str] = set()
            prev_docs: Set[str] = set()
            jaccard_values = []
            
            for idx, entry in enumerate(exploration_log, start=1):
                docs = set(entry.get("doc_ids", []))
                overlap_count = len(docs & prev_docs) if prev_docs else 0
                union_count = len(docs | prev_docs) if prev_docs else (len(docs) or 1)
                jaccard = (overlap_count / union_count) if prev_docs else 0.0
                cumulative_docs |= docs
                
                seed_per_iter.append({
                    "seed_id": seed_id,
                    "seed_iteration": idx,
                    "iteration": entry.get("iteration"),
                    "query": entry.get("query"),
                    "doc_ids": list(docs),
                    "docs_count": len(docs),
                    "overlap_with_prev": overlap_count,
                    "jaccard_with_prev": jaccard,
                    "cumulative_unique_docs": len(cumulative_docs)
                })
                
                if prev_docs:
                    jaccard_values.append(jaccard)
                prev_docs = docs
            
            seed_summary = {
                "seed_id": seed_id,
                "total_iterations": len(seed_per_iter),
                "total_unique_docs": len(cumulative_docs),
                "avg_jaccard_with_prev": sum(jaccard_values) / len(jaccard_values) if jaccard_values else 0.0
            }
            
            by_seed[seed_id] = {
                "per_iteration": seed_per_iter,
                "summary": seed_summary
            }
            
            # Build aggregated view
            all_per_iter = []
            all_docs = set()
            all_jaccards = []
            for seed_data in by_seed.values():
                for row in seed_data["per_iteration"]:
                    all_per_iter.append(row)
                    all_docs.update(row.get("doc_ids", []))
                    if row.get("seed_iteration", 0) > 1:
                        all_jaccards.append(row.get("jaccard_with_prev", 0.0))
            
            coverage["by_seed"] = by_seed
            coverage["per_iteration"] = all_per_iter
            coverage["summary"] = {
                "total_seeds": len(by_seed),
                "total_iterations": sum(len(seed_data["per_iteration"]) for seed_data in by_seed.values()),
                "total_unique_docs": len(all_docs),
                "avg_jaccard_with_prev": sum(all_jaccards) / len(all_jaccards) if all_jaccards else 0.0
            }
            
            with open(cov_file, "w") as f:
                json.dump(coverage, f, indent=2)
            logger.info(f"Saved coverage metrics: {cov_file}")
        except Exception as e:
            logger.warning(f"Failed to compute coverage metrics: {e}")

    async def _execute_query(
        self, 
        workflow, 
        query: str, 
        teacher_model: Any,
        use_kb: bool
    ) -> Dict[str, Any]:
        """Execute single query with chosen execution mode
        
        Each query starts with a fresh context (no evidence from previous queries).
        Knowledge base is injected into context metadata if use_kb is True.
        """
        
        # Determine execution mode
        mode_to_use = self.execution_mode
        
        if mode_to_use == "auto":
            # Ask teacher to choose mode
            mode_to_use = await self._choose_execution_mode(query, teacher_model)
        
        logger.info(f"Executing with mode: {mode_to_use}")
        
        # Prepare knowledge base context for injection into workflow metadata
        kb_metadata = {}
        if use_kb and len(self.knowledge_base.entries) > 0:
            kb_metadata["knowledge_base"] = self.knowledge_base.get_context()
            kb_metadata["knowledge_base_size"] = len(self.knowledge_base.entries)
            logger.info(f"Injecting knowledge base with {len(self.knowledge_base.entries)} entries into context")
        
        # Execute workflow based on mode
        # NOTE: Each query gets a fresh context (no evidence reuse between different queries)
        # Knowledge base is passed as initial_metadata so it's available in context during execution
        try:
            if mode_to_use == "standard":
                # Use StandardRunner with fresh context (no evidence from previous queries)
                from agentsim.simulation.modes.standard import StandardRunner
                runner = StandardRunner(self.template, self.clients, self.workflow_executor)
                
                # Create a sample dict for standard runner
                sample = {"query": query, "answer": ""}
                # Pass knowledge base as initial_metadata
                result = await runner.run(workflow, sample, initial_metadata=kb_metadata)
                
                # Extract evidence and doc_ids from context
                if result.get("context"):
                    ctx = result["context"]
                    evidence_store = ctx.evidence_store if hasattr(ctx, "evidence_store") else []
                    result["evidence"] = evidence_store
                    
                    # Extract unique doc IDs
                    unique_doc_ids = set()
                    for span in evidence_store:
                        if hasattr(span, 'id'):
                            unique_doc_ids.add(span.id)
                    result["doc_ids"] = list(unique_doc_ids)
                    logger.info(f"Extracted {len(result['evidence'])} evidence spans from standard mode")
                    logger.info(f"Unique document IDs: {len(result['doc_ids'])}")
                else:
                    result["evidence"] = []
                    result["doc_ids"] = []
                    logger.warning("No context in standard mode result")
                
            elif mode_to_use == "adaptive":
                # Use AdaptiveRunner with fresh context (no evidence from previous queries)
                from agentsim.simulation.modes.adaptive import AdaptiveRunner
                runner = AdaptiveRunner(self.template, self.clients, self.workflow_executor)
                
                # Create a sample dict for adaptive runner
                sample = {"query": query, "answer": ""}
                # Pass knowledge base as initial_metadata
                result = await runner.run(workflow, sample, initial_metadata=kb_metadata)
                
                # Extract evidence and doc_ids from context
                if result.get("context"):
                    ctx = result["context"]
                    evidence_store = ctx.evidence_store if hasattr(ctx, "evidence_store") else []
                    result["evidence"] = evidence_store
                    result["doc_ids"] = [span.id for span in evidence_store] if evidence_store else []
                    logger.info(f"Extracted {len(result['evidence'])} evidence spans from adaptive mode")
                    
                    # Also extract unique doc IDs
                    unique_doc_ids = set()
                    for span in evidence_store:
                        if hasattr(span, 'id'):
                            unique_doc_ids.add(span.id)
                    result["doc_ids"] = list(unique_doc_ids)
                    logger.info(f"Unique document IDs: {len(result['doc_ids'])}")
                else:
                    result["evidence"] = []
                    result["doc_ids"] = []
                    logger.warning("No context in adaptive mode result")
                
            else:
                # Fallback: execute workflow directly with fresh context
                # Pass knowledge base as metadata so it's available during execution
                workflow_context = await self.workflow_executor.execute(
                    workflow, 
                    query,
                    metadata=kb_metadata
                )
                
                result = {
                    "success": True,
                    "synthesis": workflow_context.metadata.get("final_answer") or workflow_context.metadata.get("draft_answer"),
                    "context": workflow_context,
                    "evidence": workflow_context.evidence_store,
                    "doc_ids": [span.id for span in workflow_context.evidence_store]
                }
            
            result["mode_used"] = mode_to_use
            return result
            
        except Exception as e:
            logger.error(f"Execution error: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return {
                "success": False,
                "synthesis": "",
                "evidence": [],
                "doc_ids": [],
                "similarity": 0.0,
                "mode_used": mode_to_use,
                "error": str(e)
            }
    
    async def _choose_execution_mode(self, query: str, teacher_model: Any) -> str:
        """Ask teacher model to choose execution mode"""
        
        llm = self.clients.get("llm_client")
        
        prompt = f"""Given this query, choose the best execution strategy:

Query: {query}

Modes available:
- "standard": Fixed workflow, iterative refinement until similarity threshold
- "adaptive": Dynamic component selection based on intermediate results

Which mode is best for this query? Respond with just "standard" or "adaptive"."""
        
        try:
            response = await llm.get_completion(
                prompt, 
                model=teacher_model.model_id,
                temperature=0.3
            )
            mode = response.strip().lower()
            return mode if mode in ["standard", "adaptive"] else "standard"
        except:
            return "standard"
    
    async def _reflect_and_expand(
        self, 
        query: str, 
        result: Dict[str, Any],
        teacher_model: Any
    ) -> List[str]:
        """Reflect on results and generate new exploration queries"""
        
        llm = self.clients.get("llm_client")
        
        # Build reflection context
        knowledge_context = ""
        if self.use_knowledge_base and len(self.knowledge_base.entries) > 0:
            knowledge_context = f"\n\nYour knowledge base:\n{self.knowledge_base.get_context(max_entries=3)}"
        
        # Safely get synthesis (handle None)
        synthesis = result.get('synthesis') or result.get('answer') or ''
        synthesis_text = synthesis[:500] if synthesis else 'No answer generated'
        
        prompt = f"""Reflect on this query and its answer, then generate {self.questions_per_reflection} new MSMARCO-style questions to explore.

Original query: {query}
Answer: {synthesis_text}
Evidence sources: {len(result.get('evidence', []))} documents
{knowledge_context}

CRITICAL: Generate SHORT, CONCISE questions (under 12 words each) that:
1. Explore related topics mentioned in the answer
2. Dig deeper into specific concepts or claims
3. Use natural language suitable for web search (MSMARCO-style)
4. Are answerable with factual information

Return ONLY a JSON array of exactly {self.questions_per_reflection} questions:
["short question 1", "short question 2", "short question 3"]

Example MSMARCO-style questions:
- "what causes earthquakes"
- "how do vaccines work"
- "when was the internet invented"

Your response:"""
        
        try:
            completion = await llm.get_completion(
                prompt,
                model=teacher_model.model_id,
                temperature=self.reflection_temperature,
                return_usage=True
            )
            if isinstance(completion, dict):
                response = completion.get("text", "")
                usage = completion.get("usage", {}) or {}
                tokens = usage.get("total_tokens", 0)
            else:
                response = str(completion)
                tokens = 0
            
            # Try to parse JSON
            response = response.strip()
            if response.startswith("```"):
                response = response.split("```")[1]
                if response.startswith("json"):
                    response = response[4:]
            
            new_queries = json.loads(response.strip())
            
            if isinstance(new_queries, list):
                # Add synthetic message to context for supervised training if context exists
                ctx = result.get("context")
                if ctx:
                    from agentsim.workflow.context import Message
                    ctx.add_message(Message(
                        turn=max(len(ctx.messages) + 1, 1),
                        component="reflection",
                        thought="Generated exploration questions",
                        action={"tool": "reflection", "parameters": {}},
                        observation={"questions": new_queries},
                        verdict="PROCEED",
                        latency_ms=0.0,
                        execution_time_ms=0.0,
                        llm_input=prompt,
                        llm_output=json.dumps(new_queries),
                        tokens=tokens,
                        evidence_count=len(result.get('evidence', []))
                    ))
                return [q for q in new_queries if isinstance(q, str) and q.strip()]
            return []
            
        except Exception as e:
            logger.error(f"Reflection error: {e}")
            return []
    
    async def _get_consultant_reflections(
        self,
        query: str,
        result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Get question suggestions from consultant models
        
        Returns:
            Dict with "questions" (list) and "details" (list of dicts with model info)
        """
        
        if not self.template.consultant_models:
            return {"questions": [], "details": []}
        
        llm = self.clients.get("llm_client")
        all_questions = []
        consultant_details = []
        
        # Handle case where result might be None
        if not result:
            logger.warning("No result available for consultant reflections")
            return {"questions": all_questions, "details": consultant_details}
        
        synthesis = result.get('synthesis', '') if isinstance(result, dict) else ''
        
        for consultant in self.template.consultant_models:
            prompt = f"""As a consultant model reviewing this exploration, suggest 2 additional MSMARCO-style questions to explore.

Query: {query}
Answer: {synthesis[:300] if synthesis else 'No answer available'}

CRITICAL: Generate SHORT, CONCISE questions (under 12 words each) suitable for web search.

Return ONLY a valid JSON array of strings. No explanations, no markdown, no extra text.

Format: ["short question 1", "short question 2"]

Example: ["what causes earthquakes", "how do vaccines work"]

Your response:"""
            
            try:
                completion = await llm.get_completion(
                    prompt,
                    model=consultant.model_id,
                    temperature=self.reflection_temperature,
                    return_usage=True
                )
                if isinstance(completion, dict):
                    response = completion.get("text", "")
                    usage = completion.get("usage", {}) or {}
                    tokens = usage.get("total_tokens", 0)
                else:
                    response = str(completion)
                    tokens = 0
                
                response = response.strip()
                
                # Try to extract JSON from response
                if "```json" in response:
                    # Extract from markdown code block
                    parts = response.split("```json")
                    if len(parts) > 1:
                        response = parts[1].split("```")[0].strip()
                elif "```" in response:
                    # Extract from generic code block
                    parts = response.split("```")
                    if len(parts) > 1:
                        response = parts[1].strip()
                
                # Try to find JSON array in response
                start = response.find('[')
                end = response.rfind(']')
                if start != -1 and end != -1:
                    response = response[start:end+1]
                else:
                    # If no array found, skip this consultant
                    logger.debug(f"Consultant {consultant.name} ({consultant.model_id}) did not return a valid array. Response: {response[:200]}")
                    continue
                
                questions = json.loads(response.strip())
                if isinstance(questions, list):
                    valid_questions = [q for q in questions if isinstance(q, str) and q.strip()]
                    all_questions.extend(valid_questions)
                    consultant_details.append({
                        "model": consultant.name,
                        "model_id": consultant.model_id,
                        "questions": valid_questions,
                        "count": len(valid_questions)
                    })
                    # Add synthetic message to context
                    ctx = result.get("context")
                    if ctx:
                        from agentsim.workflow.context import Message
                        ctx.add_message(Message(
                            turn=max(len(ctx.messages) + 1, 1),
                            component="consultant_reflection",
                            thought=f"Consultant {consultant.name} suggestions",
                            action={"tool": "consultant_reflection", "parameters": {"model": consultant.model_id}},
                            observation={"questions": valid_questions},
                            verdict="PROCEED",
                            latency_ms=0.0,
                            execution_time_ms=0.0,
                            llm_input=prompt,
                            llm_output=json.dumps(valid_questions),
                            tokens=tokens,
                            evidence_count=len(result.get('evidence', []))
                        ))
                    logger.info(f"✓ Consultant {consultant.name} ({consultant.model_id}) suggested {len(valid_questions)} questions")
                elif isinstance(questions, dict) and "questions" in questions:
                    # Handle {"questions": [...]} format
                    valid_questions = [q for q in questions["questions"] if isinstance(q, str) and q.strip()]
                    all_questions.extend(valid_questions)
                    consultant_details.append({
                        "model": consultant.name,
                        "model_id": consultant.model_id,
                        "questions": valid_questions,
                        "count": len(valid_questions)
                    })
                    ctx = result.get("context")
                    if ctx:
                        from agentsim.workflow.context import Message
                        ctx.add_message(Message(
                            turn=max(len(ctx.messages) + 1, 1),
                            component="consultant_reflection",
                            thought=f"Consultant {consultant.name} suggestions",
                            action={"tool": "consultant_reflection", "parameters": {"model": consultant.model_id}},
                            observation={"questions": valid_questions},
                            verdict="PROCEED",
                            latency_ms=0.0,
                            execution_time_ms=0.0,
                            llm_input=prompt,
                            llm_output=json.dumps(valid_questions),
                            tokens=tokens,
                            evidence_count=len(result.get('evidence', []))
                        ))
                    logger.info(f"✓ Consultant {consultant.name} ({consultant.model_id}) suggested {len(valid_questions)} questions")
                    
            except Exception as e:
                logger.debug(f"✗ Consultant {consultant.name} ({consultant.model_id}) reflection failed: {e}. Response: {response[:200] if 'response' in locals() else 'N/A'}")
        
        return {"questions": all_questions, "details": consultant_details}
    
    async def _verify_with_external_knowledge(
        self,
        query: str,
        answer: str,
        context: Any = None
    ) -> Dict[str, Any]:
        """Verify answer against external knowledge component
        
        This uses an LLM to generate a reference answer based on external knowledge
        for comparison with the generated answer.
        """
        
        logger.info(f"External verification for: {query}")
        
        llm = self.clients.get("llm_client")
        if not llm:
            return {
                "verified": True,
                "confidence": 0.5,
                "component_used": self.external_knowledge_component,
                "external_answer": ""
            }
        
        # Generate reference answer using external knowledge
        prompt = f"""You are an external knowledge verifier. Provide a concise, factual answer to this question based on your knowledge.

Question: {query}

Provide a brief, accurate answer (2-3 sentences):"""
        
        try:
            completion = await llm.get_completion(
                prompt,
                temperature=0.1,
                max_tokens=200,
                return_usage=True
            )
            if isinstance(completion, dict):
                external_answer = (completion.get("text") or "").strip()
                usage = completion.get("usage", {}) or {}
                tokens = usage.get("total_tokens", 0)
            else:
                external_answer = str(completion).strip()
                tokens = 0
            
            # Simple verification: if we got an answer, consider it verified
            verified = len(external_answer) > 10
            confidence = 0.85 if verified else 0.5
            # Log as synthetic message for accounting
            if context:
                from agentsim.workflow.context import Message
                context.add_message(Message(
                    turn=max(len(context.messages) + 1, 1),
                    component="external_verification",
                    thought="Generated external reference answer",
                    action={"tool": "external_verification", "parameters": {}},
                    observation={"external_answer": external_answer, "verified": verified, "confidence": confidence},
                    verdict="PROCEED",
                    latency_ms=0.0,
                    execution_time_ms=0.0,
                    llm_input=prompt,
                    llm_output=external_answer,
                    tokens=tokens,
                    evidence_count=len(context.evidence_store) if hasattr(context, "evidence_store") else 0
                ))
            
            return {
                "verified": verified,
                "confidence": confidence,
                "component_used": self.external_knowledge_component,
                "external_answer": external_answer
            }
        except Exception as e:
            logger.error(f"External verification error: {e}")
            return {
                "verified": False,
                "confidence": 0.0,
                "component_used": self.external_knowledge_component,
                "external_answer": "",
                "error": str(e)
            }
    
    async def _compute_similarity_local(self, text1: str, text2: str) -> float:
        """Compute similarity using local embeddings"""
        
        # Handle empty strings
        if not text1 or not text1.strip():
            logger.warning("Text1 is empty, returning similarity 0.0")
            return 0.0
        
        if not text2 or not text2.strip():
            logger.warning("Text2 is empty, returning similarity 0.0")
            return 0.0
        
        try:
            import numpy as np
            from sklearn.metrics.pairwise import cosine_similarity
            
            llm = self.clients.get("llm_client")
            if not llm:
                return 0.0
            
            emb1 = await llm.get_embedding(text1)
            emb2 = await llm.get_embedding(text2)
            
            sim = cosine_similarity([emb1], [emb2])[0][0]
            return float(sim)
        except Exception as e:
            logger.error(f"Similarity computation error: {e}")
            return 0.0
    
    def _save_exploration_log(self, output_path: Path, exploration_log: List[Dict]):
        """Save detailed exploration log, appending by seed and supporting resumes."""
        seed_id = self.current_seed_id or "seed"
        log_file = output_path / "exploration_log.json"
        
        # Prepare new entries ensuring seed_id present
        new_entries = []
        for entry in exploration_log:
            enriched = dict(entry)
            enriched["seed_id"] = seed_id
            new_entries.append(enriched)
        
        # Load existing log if present
        if log_file.exists():
            try:
                with open(log_file, 'r') as f:
                    data = json.load(f)
            except Exception:
                data = {}
        else:
            data = {}
        
        config_block = {
            "max_explorations": self.max_explorations,
            "use_knowledge_base": self.use_knowledge_base,
            "execution_mode": self.execution_mode
        }
        
        existing_explorations = data.get("explorations", [])
        # Remove prior entries for this seed to avoid duplicates on resume
        existing_explorations = [e for e in existing_explorations if e.get("seed_id") != seed_id]
        existing_explorations.extend(new_entries)
        
        samples_info = data.get("samples", {})
        samples_info[seed_id] = {
            "total_explorations": len(new_entries),
            "queries": [entry.get("query") for entry in new_entries],
            "documents": list({doc for entry in new_entries for doc in entry.get("doc_ids", [])})
        }
        
        # Compute summary metrics across all explorations
        unique_queries = set(e.get("query") for e in existing_explorations if e.get("query"))
        unique_docs = set()
        for e in existing_explorations:
            unique_docs.update(e.get("doc_ids", []))
        
        summary = {
            "total_explorations": len(existing_explorations),
            "total_seeds": len(samples_info),
            "total_queries": len(unique_queries),
            "total_documents": len(unique_docs)
        }
        
        payload = {
            "template_id": self.template.id,
            "mode": "exploratory",
            "config": config_block,
            "explorations": existing_explorations,
            "samples": samples_info,
            "summary": summary
        }
        
        with open(log_file, 'w') as f:
            json.dump(payload, f, indent=2)
        
        logger.info(f"Saved exploration log: {log_file}")
    
    def _save_knowledge_graph(self, output_path: Path):
        """Save knowledge graph with all queries and documents, merging across seeds."""
        seed_id = self.current_seed_id or "seed"
        
        # --- Queries explored ---
        queries_file = output_path / "queries_explored.json"
        if queries_file.exists():
            try:
                with open(queries_file, 'r') as f:
                    queries_payload = json.load(f)
            except Exception:
                queries_payload = {}
        else:
            queries_payload = {}
        
        queries_by_seed = queries_payload.get("by_seed", {})
        queries_by_seed[seed_id] = sorted(list(self.explored_queries))
        
        all_queries = set()
        for q_list in queries_by_seed.values():
            all_queries.update(q_list)
        
        queries_payload.update({
            "queries": sorted(all_queries),
            "total": len(all_queries),
            "by_seed": queries_by_seed
        })
        
        with open(queries_file, 'w') as f:
            json.dump(queries_payload, f, indent=2)
        
        # --- Documents retrieved ---
        docs_file = output_path / "documents_retrieved.json"
        if docs_file.exists():
            try:
                with open(docs_file, 'r') as f:
                    docs_payload = json.load(f)
            except Exception:
                docs_payload = {}
        else:
            docs_payload = {}
        
        documents = docs_payload.get("documents", {})
        by_seed_docs = docs_payload.get("by_seed", {})
        
        # Remove previous contribution for this seed if present
        if seed_id in by_seed_docs:
            previous_docs = by_seed_docs[seed_id]
            for doc_id, info in previous_docs.items():
                if doc_id in documents:
                    existing_queries = set(documents[doc_id].get("queries_found_in", []))
                    existing_queries -= set(info.get("queries_found_in", []))
                    new_freq = documents[doc_id].get("frequency", 0) - info.get("frequency", 0)
                    if new_freq <= 0:
                        documents.pop(doc_id, None)
                    else:
                        documents[doc_id]["queries_found_in"] = sorted(existing_queries)
                        documents[doc_id]["frequency"] = new_freq
            by_seed_docs.pop(seed_id, None)
        
        # Add new contributions for this seed
        seed_doc_data = {}
        for doc_id, queries in self.knowledge_base.doc_index.items():
            queries_list = sorted(list(queries))
            seed_doc_data[doc_id] = {
                "queries_found_in": queries_list,
                "frequency": len(queries_list)
            }
            
            doc_entry = documents.setdefault(doc_id, {"queries_found_in": [], "frequency": 0})
            updated_queries = set(doc_entry["queries_found_in"])
            updated_queries.update(queries_list)
            doc_entry["queries_found_in"] = sorted(updated_queries)
            doc_entry["frequency"] = doc_entry["frequency"] + len(queries_list)
        
        by_seed_docs[seed_id] = seed_doc_data
        docs_payload["documents"] = documents
        docs_payload["by_seed"] = by_seed_docs
        docs_payload["total_unique_documents"] = len(documents)
        docs_payload["total_retrievals"] = sum(d.get("frequency", 0) for d in documents.values())
        
        with open(docs_file, 'w') as f:
            json.dump(docs_payload, f, indent=2)
        
        # --- Knowledge base entries ---
        kb_file = output_path / "knowledge_base.json"
        if kb_file.exists():
            try:
                with open(kb_file, 'r') as f:
                    kb_payload = json.load(f)
            except Exception:
                kb_payload = {}
        else:
            kb_payload = {}
        
        all_entries = kb_payload.get("entries", [])
        # Remove old entries belonging to this seed
        all_entries = [entry for entry in all_entries if entry.get("seed_id") != seed_id]
        # Append new entries
        new_entries = [dict(entry) for entry in self.knowledge_base.entries]
        all_entries.extend(new_entries)
        
        by_seed_entries = kb_payload.get("by_seed", {})
        by_seed_entries[seed_id] = new_entries
        
        unique_docs = set()
        total_docs = 0
        unique_queries = set()
        for entry in all_entries:
            docs = entry.get("doc_ids", [])
            unique_docs.update(docs)
            total_docs += len(docs)
            unique_queries.add(entry.get("query"))
        
        kb_payload["entries"] = all_entries
        kb_payload["by_seed"] = by_seed_entries
        kb_payload["stats"] = {
            "total_entries": len(all_entries),
            "total_queries": len(unique_queries),
            "unique_documents": len(unique_docs),
            "avg_docs_per_entry": total_docs / max(len(all_entries), 1)
        }
        
        with open(kb_file, 'w') as f:
            json.dump(kb_payload, f, indent=2)
        
        logger.info(f"Saved knowledge graph artifacts to: {output_path}")
