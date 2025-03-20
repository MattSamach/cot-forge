"""
Abstract base class for all scorers.
Scorers are used to evaluate the quality and compare results
of different reasoning strategies for downselection.
This base class defines the abstract interface that all scorers must implement.
"""

from abc import ABC, abstractmethod
from typing import Any

from cot_forge.llm import LLMProvider


class BaseScorer(ABC):
    """Abstract base class for scorers."""
    
    def __init__(self,
                 name: str,
                 description: str,
                 llm_provider: LLMProvider = None,
                 llm_kwargs: dict[str, Any] | None = None,
                 **kwargs):
        self.name = name
        self.description = description
        self.llm_provider = llm_provider
        self.llm_kwargs = llm_kwargs or {}

    @abstractmethod
    def score(self,
              cot_list: list[dict[str, dict[str, Any]]],
              question: str,
              ground_truth_answer: str,
              **kwargs: Any) -> dict[str, float]:
        """Scores a list of chains of thought (CoTs) against one another.

        All strategy CoTs are passed to the scorer to provide contextualized scores.
        The scoring function should return a dictionary where the keys are the CoT names and the
        values are the scores.
        
        Args:
            cot_list: List of CoTs to be scored. Each CoT is a dictionary with a "name" and "cot" key.
                  The "name" key is the name of the strategy. The "cot" key is a dictionary
                  containing the chain of thought.
            question: The question to be answered.
            ground_truth_answer: The true answer to the question.
            llm_provider: LLM provider to use for scoring.
            llm_kwargs: Additional kwargs for LLM provider.
            **kwargs: Additional arguments.
            
        Returns:
            A dictionary where the keys are the CoT names and the values are the scores.
            
        Example:
            cot_list = [
            {
                "name": "strategy_1",
                "cot": {
                "action": "Inner Thinking", "content":...,
                ...,
                "action": "Final Answer", "content": "42"
                }
            },
            {
                "name": "strategy_2",
                "cot": {
                "action": "Inner Thinking", "content":...,
                ...,
                "action": "Final Answer", "content": "22"
                }
            },
            ...
            ]
        """
        pass
    
    def __call__(self,
                 cot_list: list[dict[str, dict[str, Any]]],
                 question: str,
                 ground_truth_answer: str,
                 **kwargs: Any) -> bool:
        """Call the score method."""
        return self.score(cot_list, question, ground_truth_answer, **kwargs)
    
    def __str__(self) -> str:
        """Return a string representation of the BaseScorer."""
        llm_info = f", LLM Provider: {self.llm_provider}" if self.llm_provider else ""
        return f"{self.name}: {self.description}" + llm_info

    def __repr__(self) -> str:
        """Return a developer-friendly string representation of the BaseScorer."""
        llm_info = f", LLM Provider: {self.llm_provider}" if self.llm_provider else ""
        return f"{self.__class__.__name__}(name='{self.name}', description='{self.description}'){llm_info})"