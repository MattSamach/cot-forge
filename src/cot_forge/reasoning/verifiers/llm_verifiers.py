"""
This module contains LLM verifiers that use a language model to verify the correctness of answers.
The LLMJudgeVerifier class is a specific implementation that uses 
an LLM to judge the correctness of an answer.
"""
import json
import logging
from typing import Any

from cot_forge.llm import LLMProvider
from cot_forge.reasoning.types import ReasoningNode
from cot_forge.reasoning.verifiers.base import BaseVerifier
from cot_forge.reasoning.verifiers.prompts import (DEFAULT_VERIFICATION_PROMPT,
                                                   VERIFICATION_FORMAT_PROMPT)
from cot_forge.utils.parsing import extract_final_answer_from_cot, parse_reasoning_response

logger = logging.getLogger(__name__)

class LLMJudgeVerifier(BaseVerifier):
    """Verifier that uses an LLM as a judge to verify answer correctness and provide feedback."""
    
    def __init__(self, prompt_template: str = DEFAULT_VERIFICATION_PROMPT):
        """Initialize with custom prompt template if desired."""
        self.prompt_template = prompt_template
        
    def build_prompt(self, final_answer: str, ground_truth_answer: str) -> str:
        """Builds the verification prompt for the LLM."""
        prompt = self.prompt_template.format(
                final_answer=final_answer,
                ground_truth_answer=ground_truth_answer
            )
        prompt += "\n\n" + VERIFICATION_FORMAT_PROMPT
        return prompt
    
    def parse_response(self, response: str) -> str:
        """Parse the LLM response to extract the final answer."""
        try:            
            response_json = parse_reasoning_response(response)
            verification_result = response_json.get("verification", {}).get("result").strip().lower()
            explanation = response_json.get("verification", {}).get("explanation")
            return verification_result, explanation
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM response: {e}")
            return False, None

    def verify(self,
               node: ReasoningNode,
               question: str,
               ground_truth_answer: str,
               llm_provider: LLMProvider,
               llm_kwargs: dict[str, Any] | None = None) -> tuple[bool, str]:
        """Use LLM to verify if the answer is correct."""
        if not node or not node.cot:
            logger.error("Node or CoT is None")
            return False, None
        
        final_answer = extract_final_answer_from_cot(node.cot)
        if final_answer is None:
            logger.warning("No Final Conclusion found in response")
            node.metadata = {
                **(node.metadata or {}),
                "warning": "missing_final_conclusion"
            }
            return False, None
        
        try:
            prompt = self.build_prompt(final_answer=final_answer, ground_truth_answer=ground_truth_answer)
            response = llm_provider.generate(
                prompt=prompt,
                **(llm_kwargs or {})
            )
            
            if not response:
                raise ValueError("Empty response from LLM")
            
            verification_result, explanation = self.parse_response(response)
            
            return verification_result == "yes", explanation 

        except Exception as e:
            logger.error(f"Error during verification: {e}")
            return False

# Create a default instance for easy importing
default_verifier = LLMJudgeVerifier()