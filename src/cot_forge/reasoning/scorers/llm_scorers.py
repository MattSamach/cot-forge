"""
This module contains LLM scorers that use a language model to 
score chains of thought (cots) against one another.
"""

import logging
from typing import Any, Literal

from cot_forge.llm import LLMProvider
from cot_forge.reasoning.scorers.base import BaseScorer
from cot_forge.reasoning.scorers.prompts import PROBABILITY_FINAL_ANSWER_PROMPT, ScorerPromptTemplate
from cot_forge.utils.parsing import extract_final_answer_from_cot, parse_json_response
from cot_forge.utils.search_utils import execute_with_fallback

logger = logging.getLogger(__name__)

class LLMScorerBase(BaseScorer):
    """Base class for all LLM-based scorers with error handling."""
    
    def generate_and_parse_scores(
        self,
        prompt: str,
        on_error: Literal["continue", "raise", "retry"] = "retry",
        max_retries: int = 3,
        retry_delay: float = 1.0
    ) -> tuple[dict, str | None]:
        """
        Generate a response from the LLM with standardized error handling.
        
        Args:
            prompt: The prompt to send to the LLM
            on_error: How to handle errors during generation
            max_retries: Maximum number of retries if on_error="retry"
            retry_delay: Delay between retries in seconds
            
        Returns:
            tuple[str, str | None]: The generated response and any error message
        """
        def helper_function():
            """Helper function to generate the response and parse the scores."""
            # Generate the response using the LLM
            response = self.llm_provider.generate(
                prompt=prompt,
                **(self.llm_kwargs)
            )
            # Parse the response by extracting the JSON content
            scores = parse_json_response(response)["scoring"]
            
            return scores
            
        result, error_msg = execute_with_fallback(
            operation_name="LLM generation for scoring",
            operation_func=helper_function,
            on_error=on_error,
            max_retries=max_retries,
            retry_delay=retry_delay,
            logger=logger,
            fallback_value=None
        )
        
        if error_msg and (on_error == "raise" or on_error == "retry"):
            logger.error(f"LLM generation for scoring failed: {error_msg}")
            raise RuntimeError(f"LLM generation for scoring failed: {error_msg}")
            
        return result, error_msg

class ProbabilityFinalAnswerScorer(LLMScorerBase):
    """Scorer that only uses the final answer to score the CoT and gives scores
    in the form of a "probability" of the final answer leading to the ground truth answer."""
    
    def __init__(self,
                 llm_provider: LLMProvider,
                 llm_kwargs = None,
                 **kwargs):
        """Initialize with the LLM provider and any additional kwargs."""
        name = "probability_final_answer_scorer"
        description = "Scorer gives probability scores for the final answer of each strategy's CoT."
        super().__init__(name, description, llm_provider, llm_kwargs, **kwargs)

    def score(self,
              cot_list: list[dict[str, dict[str, Any]]],
              question: str,
              ground_truth_answer: str,
              **kwargs: Any) -> dict[str, float]:
        
        try:
            final_answers = [
                {
                    "strategy_name": cot["strategy_name"],
                    "final_answer": extract_final_answer_from_cot(cot["cot"])
                }
                for cot in cot_list
            ]
        except Exception as e:
            logger.error(f"Failed to extract final answers from CoTs: {e}")
            return {}
        
        # Format the final answers into a string
        final_answers_formatted = '\n'.join([
            f"{item['strategy_name']}: {item['final_answer']},"
            for item in final_answers
        ])
        
        # Generate the prompt
        prompt = ScorerPromptTemplate.build_prompt(
            question=question,
            answer=ground_truth_answer,
            instruction_prompt=PROBABILITY_FINAL_ANSWER_PROMPT,
            final_answers=final_answers_formatted
        )
        
        # Generate the response and parse the scores
        scores, error_msg = self.generate_and_parse_scores(
            prompt=prompt,
            on_error="retry",
            max_retries=3,
            retry_delay=1.0
        )
        
        if error_msg:
            logger.error(f"Failed to generate scores: {error_msg}")
            
        return {k: float(v) for k, v in scores.items()}

                
        # try:
        #     response = self.llm_provider.generate(
        #         prompt=prompt,
        #         **(self.llm_kwargs)
        #     )
        # except Exception as e:
        #     logger.error(f"Error in generating response: {e}")
        #     return {}
        
        # Parse the response by extracting the JSON content
        # try:
        #     scores = parse_json_response(response)["scoring"]
            
        # except json.JSONDecodeError as e:
        #     logger.error(f"Failed to parse LLM response: {e}")
        #     return {}
        
        # Convert scores to float and return
