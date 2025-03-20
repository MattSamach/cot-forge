"""
LLM-based verification implementations for the CoT-Forge reasoning framework.

This module provides verifier implementations that leverage Language Models (LLMs)
to assess the correctness of reasoning outputs. The primary class, LLMJudgeVerifier,
uses an LLM to compare and evaluate answers against ground truth.

Key Features:
    - LLM-based verification of reasoning outputs
    - Configurable prompt templates for verification
    - JSON-formatted response parsing
    - Detailed error handling and logging

Example:
    ```python
    from cot_forge.llm import OpenAIProvider
    from cot_forge.reasoning.verifiers import LLMJudgeVerifier

    # Initialize verifier
    llm = OpenAIProvider(api_key="your-key")
    verifier = LLMJudgeVerifier(
        llm_provider=llm,
        llm_kwargs={"temperature": 0.1}
    )

    # Verify a reasoning node
    is_correct, explanation = verifier.verify(
        node=reasoning_node,
        question="What is the capital of France?",
        ground_truth_answer="Paris"
    )
    ```
"""

import json
import logging
from typing import Any

from cot_forge.llm import LLMProvider
from cot_forge.reasoning.types import ReasoningNode
from cot_forge.reasoning.verifiers.base import BaseVerifier
from cot_forge.reasoning.verifiers.prompts import DEFAULT_VERIFICATION_PROMPT, VERIFICATION_FORMAT_PROMPT
from cot_forge.utils.parsing import extract_final_answer_from_cot, parse_json_response

logger = logging.getLogger(__name__)

class LLMJudgeVerifier(BaseVerifier):
    """
    A verifier that uses an LLM to judge answer correctness and provide feedback.
    
    This verifier delegates the verification task to an LLM, which compares a final
    answer against a ground truth answer and provides both a binary correctness
    assessment and a detailed explanation.

    Attributes:
        prompt_template (str): Template string for formatting verification prompts
        llm_provider (LLMProvider): Provider for LLM access
        llm_kwargs (dict): Additional parameters for LLM calls
    """
    
    def __init__(self,
                 llm_provider: LLMProvider,
                 llm_kwargs: dict[str, Any] | None = None,
                 prompt_template: str = DEFAULT_VERIFICATION_PROMPT):
        """
        Initialize the LLM judge verifier.

        Args:
            llm_provider (LLMProvider): Provider for LLM access
            llm_kwargs (dict, optional): Additional parameters for LLM calls
            prompt_template (str, optional): Custom prompt template for verification.
                Defaults to DEFAULT_VERIFICATION_PROMPT.
        """
        name = "llm_judge_verifier",
        description = "A basic LLM judge verifier that compares a final answer with a ground truth answer.",
        super().__init__(name=name, description=description, llm_provider=llm_provider, llm_kwargs=llm_kwargs)
        self.prompt_template = prompt_template
        
    def build_prompt(self, final_answer: str, ground_truth_answer: str) -> str:
        """
        Build the verification prompt for the LLM.

        Args:
            final_answer (str): The answer to verify
            ground_truth_answer (str): The correct answer to compare against

        Returns:
            str: Formatted prompt string for the LLM
        """
        prompt = self.prompt_template.format(
                final_answer=final_answer,
                ground_truth_answer=ground_truth_answer
            )
        prompt += "\n\n" + VERIFICATION_FORMAT_PROMPT
        return prompt
    
    def parse_response(self, response: str) -> tuple[str, str]:
        """
        Parse the LLM's verification response.

        Args:
            response (str): Raw response from the LLM

        Returns:
            tuple[str, str]: A tuple containing:
                - verification result ("yes"/"no")
                - explanation of the verification decision

        Raises:
            json.JSONDecodeError: If response cannot be parsed as JSON
        """
        try:            
            response_json = parse_json_response(response)
            verification_result = response_json.get("verification", {}).get("result").strip().lower()
            explanation = response_json.get("verification", {}).get("explanation")
            return verification_result, explanation
        except Exception as e:
            logger.error(f"Failed to parse LLM response: {e}")
            return False, f"Error: {str(e)}"

    def verify(self,
               node: ReasoningNode,
               question: str,
               ground_truth_answer: str) -> tuple[bool, str]:
        """
        Verify the correctness of a reasoning node's answer using an LLM.

        Args:
            node (ReasoningNode): The reasoning node to verify
            question (str): The original question being answered
            ground_truth_answer (str): The correct answer to compare against

        Returns:
            tuple[bool, str]: A tuple containing:
                - boolean indicating if the answer is correct
                - explanation of the verification result

        Raises:
            ValueError: If LLM returns an empty response
        """
        if not node.cot:
            logger.error("Node.cot is None")
            return False, "Error: Node.cot is None"
        
        final_answer = extract_final_answer_from_cot(node.cot)
        if final_answer is None:
            logger.warning("No Final Conclusion found in response")
            node.metadata = {
                **(node.metadata or {}),
                "warning": "missing_final_conclusion"
            }
            return False, "Error: No Final Conclusion found in response"
        
        try:
            prompt = self.build_prompt(final_answer=final_answer, ground_truth_answer=ground_truth_answer)
            response = self.llm_provider.generate(
                prompt=prompt,
                llm_kwargs=self.llm_kwargs
            )
            
            if not response:
                raise ValueError("Empty response from LLM")
            
            verification_result, explanation = self.parse_response(response)
            
            return verification_result == "yes", explanation 

        except Exception as e:
            logger.error(f"Error during verification: {e}")
            return False, f"Error: {str(e)}"