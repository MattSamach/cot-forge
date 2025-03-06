# src/cot_forge/reasoning/verifiers/llm_verifiers.py
import logging
from typing import Any

from cot_forge.llm import LLMProvider
from cot_forge.reasoning.types import ReasoningNode
from cot_forge.reasoning.verifiers.base import BaseVerifier
from cot_forge.reasoning.verifiers.prompts import DEFAULT_VERIFICATION_PROMPT
from cot_forge.utils.parsing import extract_final_answer_from_cot

logger = logging.getLogger(__name__)

class LLMJudgeVerifier(BaseVerifier):
    """Verifier that uses an LLM as a judge to verify answer correctness."""
    
    def __init__(self, prompt_template: str = DEFAULT_VERIFICATION_PROMPT):
        """Initialize with custom prompt template if desired."""
        self.prompt_template = prompt_template
    
    def verify(self, 
               node: ReasoningNode,
               question: str,
               ground_truth_answer: str,
               llm_provider: LLMProvider,
               llm_kwargs: dict[str, Any] | None = None) -> bool:
        """Use LLM to verify if the answer is correct."""
        if not node:
            return False
        
        final_answer = extract_final_answer_from_cot(node.cot)
        if not final_answer:
            return False
        
        try:
            prompt = self.prompt_template.format(
                final_answer=final_answer,
                ground_truth_answer=ground_truth_answer
            )
            response = llm_provider.generate(
                prompt=prompt,
                **(llm_kwargs or {})
            )
            response = response.strip().lower()
            
            if not response:
                raise ValueError("Empty response from LLM")
            
            return response == "yes"
        
        except Exception as e:
            logger.error(f"Error during verification: {e}")
            return False

# Create a default instance for easy importing
default_verifier = LLMJudgeVerifier()