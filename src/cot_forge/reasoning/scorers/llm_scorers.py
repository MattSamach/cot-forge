"""
This module contains LLM scorers that use a language model to 
score chains of thought (cots) against one another.
"""

import logging
from typing import Any, Literal

from cot_forge.llm import LLMProvider
from cot_forge.reasoning.scorers.base import BaseScorer
from cot_forge.reasoning.scorers.prompts import PROBABILITY_FINAL_ANSWER_PROMPT, ScorerPromptTemplate
from cot_forge.utils.parsing import extract_final_answer_from_cot
from cot_forge.utils.search_utils import generate_and_parse_json

logger = logging.getLogger(__name__)

class ProbabilityFinalAnswerScorer(BaseScorer):
    """Scorer that only uses the final answer to score the CoT and gives scores
    in the form of a "probability" of the final answer leading to the ground truth answer."""
    
    def __init__(self,
                 llm_provider: LLMProvider,
                 llm_kwargs = None,
                 **kwargs):
        """Initialize with the LLM provider and any additional kwargs."""
        name = "Probability Final Answer Scorer"
        description = "Scorer gives probability scores [0.0-1.0] for the final answer of each strategy's CoT."
        super().__init__(name, description, llm_provider, llm_kwargs, **kwargs)

    def generate_and_parse_scores(
        self,
        prompt: str,
        on_error: Literal["continue", "raise", "retry"] = "retry",
        max_retries: int = 3,
        retry_delay: float = 1.0
    ) -> tuple[dict, str | None]:
        """Generate and parse scores using the LLM provider."""
        return generate_and_parse_json(
            generation_llm=self.llm_provider,
            prompt=prompt,
            on_error=on_error,
            llm_kwargs=self.llm_kwargs,
            max_retries=max_retries,
            retry_delay=retry_delay,
            logger=logger,
            retrieval_object="scoring"
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> 'ProbabilityFinalAnswerScorer':
        """Create a verifier instance from a dictionary representation."""
        if not data.get("llm_provider"):
            raise ValueError("Missing llm_provider in data")
        llm_provider = LLMProvider.from_dict(data["llm_provider"])
        llm_kwargs = data.get("llm_kwargs", {})
        return cls(
            llm_provider=llm_provider,
            llm_kwargs=llm_kwargs
        )

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
        response, scores = self.generate_and_parse_scores(
            prompt=prompt,
            on_error="retry",
            max_retries=3,
            retry_delay=1.0
        )

        if not scores or not response:
            logger.error("Failed to generate scores")
            
        return {k: float(v) for k, v in scores.items()}
