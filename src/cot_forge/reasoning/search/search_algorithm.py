"""
This module defines the core interfaces and abstract base classes for implementing search algorithms 
within the cot-forge framework. It provides a flexible and extensible way to explore different 
reasoning paths using Language Model (LLM) providers, verifiers, and scorers.

The module introduces the `SearchAlgorithm` protocol, which outlines the required interface for any 
search algorithm.  The `BaseSearch` abstract class provides a foundation for concrete search 
algorithm implementations, handling common tasks and enforcing the structure defined by the protocol.

Key components:

- `SearchAlgorithm`: A protocol that defines the `_search` method, the entry point for executing a search 
    algorithm.

- `BaseSearch`: An abstract base class implementing the `SearchAlgorithm` protocol. It provides a `__call__` 
    method that serves as the main entry point, allowing subclasses to add pre- and post-processing logic 
    around the core `_search` method.  Subclasses must implement the `_search` method to define the specific 
    search strategy.

The module also leverages other components from the `cot-forge` library, such as:

- `LLMProvider`:  An interface for interacting with different Language Model providers (e.g., OpenAI, Anthropic).
- `BaseVerifier`: An abstract base class for verifying the correctness or quality of generated reasoning steps.
- `BaseScorer`: An abstract base class for scoring reasoning paths based on various criteria.
- `StrategyRegistry`: A registry for managing and accessing different reasoning strategies.

Usage:

To implement a custom search algorithm, create a class that inherits from `BaseSearch` and implements 
the `_search` method.  The `_search` method should take a question, ground truth answer, LLM provider, 
verifier, scorer, strategy registry, and optional keyword arguments as input. It should then use these 
components to explore different reasoning paths and return a `SearchResult` object containing the best 
reasoning path found.
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Protocol

from cot_forge.llm import LLMProvider
from cot_forge.reasoning.scorers import BaseScorer
from cot_forge.reasoning.strategies import (StrategyRegistry,
                                            default_strategy_registry)
from cot_forge.reasoning.types import SearchResult
from cot_forge.reasoning.verifiers import BaseVerifier

logger = logging.getLogger(__name__)

#TODO: Separate llmproviders for search, verifiers, scorers, etc...
#TODO: Create StrategySelector as its own class
#TODO: Create wrapper function for generation/cot extraction with retry logic

class SearchAlgorithm(Protocol):
    """Protocol defining the interface for search algorithms."""
    def _search(
        self,
        question: str,
        ground_truth_answer: str,
        reasoning_llm: LLMProvider,
        verifier: BaseVerifier,
        scorer: BaseScorer = None,
        strategy_registry: StrategyRegistry = default_strategy_registry, 
        llm_kwargs: dict[str, Any] = None,
        **kwargs
    ) -> SearchResult:
        ...

class BaseSearch(ABC, SearchAlgorithm):
    """
    Base class providing common functionality for search algorithms.
    Implements the SearchAlgorithm protocol using __call__ and _search.
    """

    def __call__(
        self, 
        question: str, 
        ground_truth_answer: str,
        reasoning_llm: LLMProvider,
        verifier: BaseVerifier, 
        scorer: BaseScorer = None,
        strategy_registry: StrategyRegistry = default_strategy_registry, 
        llm_kwargs: dict[str, Any] = None,
        **kwargs
    ) -> SearchResult:
        """
        Entry point matching the SearchAlgorithm protocol. Subclasses can 
        invoke additional common logic here before or after _search.
        """
        # Common initialization/validation logic (if needed)
        return self._search(
            question = question, 
            ground_truth_answer = ground_truth_answer,
            reasoning_llm = reasoning_llm,
            verifier = verifier,
            scorer = scorer,
            strategy_registry = strategy_registry,
            llm_kwargs = llm_kwargs,
            **kwargs
        )

    @abstractmethod
    def _search(
        self,
        question: str,
        ground_truth_answer: str,
        reasoning_llm: LLMProvider,
        verifier: BaseVerifier,
        scorer: BaseScorer = None,
        strategy_registry: StrategyRegistry = default_strategy_registry,
        llm_kwargs: dict[str, Any] = None,
        **kwargs
    ) -> SearchResult:
        """
        Child classes must implement the actual search logic here.
        """
        pass