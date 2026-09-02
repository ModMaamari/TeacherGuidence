"""Real-time verification of synthesis against retrieved documents"""

from typing import List, Dict, Any
from loguru import logger


class RealtimeVerifier:
    """Verifies synthesis claims against retrieved evidence"""
    
    def __init__(self, verifier_llm, config, model_id: str):
        self.llm = verifier_llm
        self.config = config
        self.model_id = model_id
    
    async def verify_synthesis(
        self,
        synthesis: str,
        evidence_spans: List[Any],
        context: Any
    ) -> Dict[str, Any]:
        """Verify synthesis against evidence"""
        
        if not self.config.enabled or not self.config.check_synthesis:
            return {"verified": True, "flagged": False}
        
        # Check if synthesis is a "cannot answer" response
        cannot_answer_phrases = [
            "cannot answer",
            "cannot provide",
            "insufficient evidence",
            "not enough information",
            "no information",
            "unable to answer",
            "don't have enough",
            "cannot determine",
            "not provided in the evidence",
            "evidence does not contain",
            "evidence doesn't contain"
        ]
        
        synthesis_lower = synthesis.lower()
        is_cannot_answer = any(phrase in synthesis_lower for phrase in cannot_answer_phrases)
        
        if is_cannot_answer:
            logger.info("Synthesis indicates insufficient evidence - skipping hallucination check")
            return {
                "verified": True,
                "flagged": False,
                "reason": "Valid 'cannot answer' response - no hallucination check needed",
                "hallucinated_claims": []
            }
        
        # Extract evidence text
        evidence_text = "\n\n".join([
            f"[{i+1}] {span.text[:200]}"
            for i, span in enumerate(evidence_spans[:5])
        ])
        
        # Verify with LLM
        prompt = f"""You are a verification system. Your ONLY job is to check if claims in the answer are supported by the provided evidence.

SYNTHESIS:
{synthesis}

EVIDENCE:
{evidence_text}

CRITICAL RULES:
1. If the synthesis says "I cannot answer" or similar, return {{"verified": true}} - this is NOT a hallucination
2. Only flag as hallucination if the synthesis makes SPECIFIC CLAIMS that are NOT in the evidence
3. General statements without specific claims should be verified as true

Return ONLY valid JSON in this EXACT format (no extra text):
{{"verified": true, "reason": "explanation", "hallucinated_claims": []}}

Set "verified" to false ONLY if there are specific factual claims not supported by evidence.
JSON response:"""
        
        try:
            response = await self.llm.get_completion(
                prompt,
                temperature=0.0,
                model=self.model_id,
                max_tokens=300
            )
            
            import json
            # Try to extract JSON if response has extra text
            response = response.strip()
            if not response.startswith('{'):
                # Find first { and last }
                start = response.find('{')
                end = response.rfind('}')
                if start != -1 and end != -1:
                    response = response[start:end+1]
            
            result = json.loads(response)
            
            # Flag if hallucination detected
            flagged = not result.get("verified", True)
            
            if flagged:
                logger.warning(f"HALLUCINATION DETECTED: {result.get('reason')}")
            
            return {
                "verified": result.get("verified", True),
                "flagged": flagged,
                "reason": result.get("reason", ""),
                "hallucinated_claims": result.get("hallucinated_claims", [])
            }
            
        except Exception as e:
            logger.error(f"Verification error: {e}")
            return {"verified": False, "flagged": True, "error": str(e)}

