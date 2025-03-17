"""
This module contains LLM scorers that use a language model to 
score chains of thought (cots) against one another.
"""

import json
import logging
from typing import Any

from cot_forge.llm import LLMProvider
from cot_forge.reasoning.scorers.base import BaseScorer
from cot_forge.reasoning.scorers.prompts import (
    PROBABILITY_FINAL_ANSWER_PROMPT, ScorerPromptTemplate)
from cot_forge.utils.parsing import (extract_final_answer_from_cot,
                                     parse_json_response)

logger = logging.getLogger(__name__)

class ProbabilityFinalAnswerScorer(BaseScorer):
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
        except:
            logger.error("Failed to extract final answers from CoTs")
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
        
        # Generate the response
        try:
            response = self.llm_provider.generate(
                prompt=prompt,
                **(self.llm_kwargs)
            )
        except Exception as e:
            logger.error(f"Error in generating response: {e}")
            return {}
        
        # Parse the response by extracting the JSON content
        try:
            scores = parse_json_response(response)["scoring"]
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM response: {e}")
            return {}
        
        # Convert scores to float and return
        return {k: float(v) for k, v in scores.items()}
