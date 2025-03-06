"""
This file defines the interface for search algorithms 
used in the chain of thought reasoning process. To create a new search algorithm,
you must define a class that implements the SearchAlgorithm protocol, including 
a StrategySelector and BaseVerifier. See `naive_linear_search.py` for an example.
"""

import logging
from typing import Any, Protocol

from cot_forge.llm import LLMProvider
from cot_forge.reasoning.strategies import StrategyRegistry, default_strategy_registry
from cot_forge.reasoning.types import SearchResult
from cot_forge.reasoning.verifiers import BaseVerifier

logger = logging.getLogger(__name__)

class SearchAlgorithm(Protocol):
    """Protocol defining the interface for search algorithms."""
    def __call__(
        self,
        question: str,
        ground_truth_answer: str,
        llm_provider: LLMProvider,
        verifier: BaseVerifier,
        strategy_registry: StrategyRegistry = default_strategy_registry, 
        llm_kwargs: dict[str, Any] | None = None,
        **kwargs
    ) -> SearchResult:
        """
        Execute a search algorithm to generate a chain of thought.
        
        Args:
            question: The question to answer.
            ground_truth_answer: The true answer to the question.
            llm_provider: An instance of LLM provider to use for generation.
            verifier: An instance of a verifier to check if the answer is correct.
            strategy_selector: Function to select the next strategy.
            chain_initializer: Function to initialize the chain of thought.
            strategy_registry: Registry of available strategies.
            llm_kwargs: Additional kwargs for the LLM provider.
            **kwargs: Additional kwargs.
        
        Returns:
            A SearchResult containing the chain of thought and metadata.
        """
        ...
