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

- `LLMProvider`:  An interface for interacting with different Language Model providers (e.g., OpenAI, Gemini).
- `BaseVerifier`: An abstract base class to verify the correctness or quality of generated reasoning steps.
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
from typing import Any, Literal, Protocol

from cot_forge.llm import LLMProvider
from cot_forge.reasoning.scorers import BaseScorer
from cot_forge.reasoning.strategies import Strategy, StrategyRegistry, default_strategy_registry
from cot_forge.reasoning.types import ReasoningNode, SearchResult
from cot_forge.reasoning.verifiers import BaseVerifier
from cot_forge.utils.search_utils import execute_with_fallback

logger = logging.getLogger(__name__)

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

    This class implements the `SearchAlgorithm` protocol and provides a foundation for
    concrete search algorithm implementations. Subclasses must implement the `_search`
    method to define the specific search strategy.

    Key Features:
    - Implements the `__call__` method to serve as the main entry point for executing a search.
    - Provides utility methods like `verify_node` and `create_node` to standardize common tasks.
    - Enforces the structure defined by the `SearchAlgorithm` protocol.

    Attributes:
        logger (logging.Logger): Logger instance for logging search-related events.

    Usage:
        To create a custom search algorithm, inherit from `BaseSearch` and implement the `_search` method.
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
        Entry point for executing a search algorithm.

        This method matches the `SearchAlgorithm` protocol and invokes the `_search` method
        to perform the actual search. Subclasses can override this method to add pre- or
        post-processing logic around the core `_search` method.

        Args:
            question (str): The question to answer.
            ground_truth_answer (str): The true answer to the question.
            reasoning_llm (LLMProvider): The LLM provider used to generate reasoning steps.
            verifier (BaseVerifier): The verifier used to check the correctness of the reasoning steps.
            scorer (BaseScorer, optional): The scorer used to evaluate reasoning paths.
            strategy_registry (StrategyRegistry, optional): The registry of reasoning strategies.
            llm_kwargs (dict[str, Any], optional): Additional keyword arguments for the LLM provider.
            **kwargs: Additional keyword arguments for the search algorithm.

        Returns:
            SearchResult: The result of the search, including the best reasoning path found.

        Raises:
            Exception: If an error occurs during the search process.
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
    
    @abstractmethod
    def to_dict(self) -> dict[str, Any]:
        """
        Convert the search algorithm to a dictionary representation.
        
        This method should be implemented by subclasses to provide a 
        reproducible representation of the search algorithm.
        """
        pass
    
    @classmethod
    @abstractmethod
    def from_dict(cls, config: dict[str, Any]) -> None:
        """
        Load the search algorithm from a dictionary representation.
        
        This method should be implemented by subclasses to restore the 
        state of the search algorithm from a dictionary.
        """
        pass
    
    def verify_node(
        self,
        node: ReasoningNode, 
        question: str, 
        ground_truth_answer: str, 
        verifier: BaseVerifier,
        on_error: Literal["continue", "raise", "retry"] = "retry",
        max_retries: int = 3,
        retry_delay: float = 1.0,
        logger: logging.Logger = logger
    ) -> tuple[bool, str | None]:
        """
        Verify a reasoning node and optionally update its status.

        This method uses the provided verifier to check the correctness of a reasoning node.
        It supports configurable error-handling strategies, including retries for transient errors.

        Args:
            node (ReasoningNode): The reasoning node to verify.
            question (str): The original question being answered.
            ground_truth_answer (str): The true answer to the question.
            verifier (BaseVerifier): The verifier used to check the correctness of the node.
            on_error (Literal["continue", "raise", "retry"], optional): How to handle verification errors:
                - "continue": Log the error and return False without raising an exception.
                - "raise": Raise an exception if verification fails.
                - "retry": Retry the verification up to `max_retries` times. Defaults to "retry".
            max_retries (int, optional): Maximum number of retry attempts if `on_error="retry"`.
                Defaults to 3.
            retry_delay (float, optional): Seconds to wait between retry attempts. Defaults to 1.0.
            logger (logging.Logger, optional): Logger instance for logging errors and retries.
                Defaults to the module logger.

        Returns:
            tuple[bool, str | None]: A tuple containing:
                - A boolean indicating whether the verification was successful.
                - An error message, if any.

        Raises:
            RuntimeError: If verification fails and `on_error="raise" or on_error="retry"`.
        """
        
        result, error_msg = execute_with_fallback(
            operation_name="verification",
            operation_func=verifier,
            args=(node, question, ground_truth_answer),
            on_error=on_error,
            max_retries=max_retries,
            retry_delay=retry_delay,
            fallback_value=None,
            logger=logger
        )
        
        if error_msg and (on_error == "raise" or on_error == "retry"):
            # Log the error and raise an exception
            logger.error(f"Verification call failed: {error_msg}")
            raise RuntimeError(f"Verification call failed: {error_msg}")
        
        elif error_msg and on_error == "continue":
            # Log the error but continue
            logger.error(f"Verification call failed: {error_msg}")
            node.metadata["verification_error"] = error_msg
            node.success = False
            node.is_final = False
            return False, error_msg
        
        # Handle the case where the verification call returned None
        if result is None:
            logger.error("Verification call returned None")
            node.metadata["verification_error"] = "Verification call returned None"
            node.success = False
            node.is_final = False
            return False, "None result from verification call"
        
        # Verification call was successful
        verification_result, explanation = result
        node.metadata["verification"] = explanation

        if verification_result:
            # Verification was successful
            node.success = True
            node.is_final = True

        return verification_result, None
    
    def create_node(
        self,
        strategy: Strategy,
        prompt: str,
        response: str = None,
        cot: list[dict[str, str]] = None,
        parent: ReasoningNode = None,
        metadata: dict[str, Any] = None,
        **kwargs: Any
    ) -> ReasoningNode:
        """
        Create a new reasoning node with standardized initialization and handling.
        Args:
            strategy: The strategy used to generate this node
            prompt: The prompt used to generate the response
            response: The response from the LLM (may be None if not yet generated)
            cot: The extracted chain of thought (may be None if not yet extracted)
            parent: The parent node (may be None if root node)
            metadata: Optional metadata dictionary for the node
            **kwargs: Additional attributes to add to the node
        Returns:
            ReasoningNode: A new reasoning node
        """
        # Create the node
        node = ReasoningNode(
            strategy=strategy,
            prompt=prompt,
            response=response or "",
            cot=cot,
            parent=parent,
            metadata=metadata or {}
        )
        
        # Set up parent-child relationship if parent exists
        if parent:
            parent.add_child(node)
        
        return node
    
    def __str__(self) -> str:
        """
        Return a string representation of the search algorithm.
        
        Returns:
            str: The class name of the search algorithm.
        """
        return self.__class__.__name__

    def __repr__(self) -> str:
        """
        Return a detailed string representation of the search algorithm.
        
        This representation includes any configuration parameters and should
        be detailed enough to recreate the object.
        
        Returns:
            str: A string in the format "ClassName(param1=value1, param2=value2)"
        """
        # Get instance attributes excluding private ones (those starting with '_')
        params = ", ".join(f"{k}={repr(v)}" for k, v in self.__dict__.items() 
                        if not k.startswith('_'))
        return f"{self.__class__.__name__}({params})"
            