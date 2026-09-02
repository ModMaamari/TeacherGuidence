"""Execute components with parallel consultant models"""

import asyncio
from typing import List, Dict, Any
from loguru import logger


class ParallelExecutor:
    """Execute components with multiple models in parallel"""
    
    def __init__(self, consultant_clients: Dict[str, Any]):
        self.consultant_clients = consultant_clients
    
    async def execute_parallel(
        self,
        component_type: str,
        consultant_models: List[str],
        teacher_result: Any,
        context: Any
    ) -> Dict[str, Any]:
        """Execute component with consultant models in parallel"""
        
        results = {"teacher": teacher_result}
        
        # Run consultants in parallel
        tasks = []
        for model_name in consultant_models:
            if model_name in self.consultant_clients:
                task = self._execute_consultant(
                    model_name,
                    component_type,
                    context
                )
                tasks.append((model_name, task))
        
        # Wait for all consultants
        for model_name, task in tasks:
            try:
                result = await task
                results[f"consultant_{model_name}"] = result
            except Exception as e:
                logger.error(f"Consultant {model_name} failed: {e}")
                results[f"consultant_{model_name}"] = {"error": str(e)}
        
        return results
    
    async def _execute_consultant(
        self,
        model_name: str,
        component_type: str,
        context: Any
    ) -> Dict[str, Any]:
        """Execute single consultant model"""
        
        llm_client = self.consultant_clients[model_name]
        
        # For synthesis components
        if component_type in ["synthesis", "answer_drafter"]:
            evidence_text = "\n\n".join([
                f"[{i+1}] {span.text[:300]}"
                for i, span in enumerate(context.evidence_store[:10])
            ])
            
            prompt = f"""Answer based ONLY on provided evidence.

Query: {context.query}

Evidence:
{evidence_text}

Answer:"""
            
            response = await llm_client.get_completion(prompt, temperature=0.1)
            return {"answer": response, "model": model_name}
        
        return {"skipped": True}

