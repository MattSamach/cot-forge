"""
This file defines the interface for search algorithms 
used in the chain of thought reasoning process. To create a new search algorithm,
you must define a class that implements the SearchAlgorithm protocol, including 
a StrategySelector and TerminationChecker. See `naive_search.py` for an example.
"""

import logging
from typing import Any, Optional, Protocol, TypedDict

from cot_forge.llm import LLMProvider
from cot_forge.reasoning.strategies import Strategy, StrategyRegistry, default_strategy_registry

logger = logging.getLogger(__name__)

class ReasoningStep(TypedDict):
    """Represents a step in the chain of thought reasoning process."""
    strategy: Strategy
    prompt: str
    response: str
    is_final: bool

class SearchResult(TypedDict):
    """Represents the result of a search algorithm."""
    chain: list[ReasoningStep]
    success: bool
    final_answer: Optional[str]
    metadata: dict[str, Any]

class StrategySelector(Protocol):
    """Protocol for strategy selection functions."""
    def __call__(self, 
                 question: str, 
                 chain: list[ReasoningStep], 
                 registry: StrategyRegistry) -> Strategy:
        """Select the next strategy to use."""
        ...

class TerminationChecker(Protocol):
    """Protocol for termination checking functions."""
    def __call__(self, 
                chain: list[ReasoningStep],
                question: str,
                ground_truth_answer: str,
                llm_provider: LLMProvider,
                llm_kwargs: dict[str, Any] | None = None
                ) -> bool:
        """Check if the search should terminate."""
        ...

class SearchAlgorithm(Protocol):
    """Protocol defining the interface for search algorithms."""
    def __call__(
        self,
        question: str,
        ground_truth_answer: str,
        llm_provider: LLMProvider,
        termination_checker: TerminationChecker,
        strategy_selector: StrategySelector,
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
            termination_checker: Function to check if search should terminate.
            strategy_selector: Function to select the next strategy.
            chain_initializer: Function to initialize the chain of thought.
            strategy_registry: Registry of available strategies.
            llm_kwargs: Additional kwargs for the LLM provider.
            **kwargs: Additional kwargs.
        
        Returns:
            A SearchResult containing the chain of thought and metadata.
        """
        ...
