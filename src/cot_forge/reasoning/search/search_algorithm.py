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
import time
from abc import ABC, abstractmethod
from typing import Any, Literal, Protocol

from cot_forge.llm import LLMProvider
from cot_forge.reasoning.scorers import BaseScorer
from cot_forge.reasoning.strategies import (Strategy, StrategyRegistry,
                                            default_strategy_registry)
from cot_forge.reasoning.types import ReasoningNode, SearchResult
from cot_forge.reasoning.verifiers import BaseVerifier
from cot_forge.utils.parsing import extract_cot
from cot_forge.utils.search_utils import execute_with_fallback

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
    
    def create_node(
        self,
        strategy: Strategy,
        prompt: str,
        response:str = None,
        cot: list[dict[str, str]] = None,
        parent: ReasoningNode = None,
        metadata: dict[str, Any] = None,
        **kwargs
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
    
    def generate_and_parse_cot(
        self,
        reasoning_llm: LLMProvider,
        prompt: str,
        llm_kwargs: dict[str, Any] = None,
        on_error: Literal["continue", "raise", "retry"] = "retry",
        max_retries: int = 3,
        retry_delay: float = 1.0,
        logger: logging.Logger = logger
    ) -> tuple[str, list[dict[str, str]]]:
        """
        Generate and parse the chain of thought (CoT) from the LLM.
        
        Args:
            reasoning_llm: The LLM provider to use for generation
            prompt: The prompt to send to the LLM
            llm_kwargs: Additional kwargs for LLM generation
            on_error: How to handle errors during generation
            max_retries: Maximum number of retries if on_error="retry"
            retry_delay: Delay between retries in seconds
        Returns:
            tuple[str, list[dict[str, str]]]: The generated response and parsed CoT
        """
        
        def helper_function():
            # Generate the response using the LLM
            response = reasoning_llm.generate(prompt, **(llm_kwargs or {}))
            # Extract the CoT from the response
            cot = extract_cot(response)
            return response, cot
        
        # Execute the operation with error handling
        result, error_msg = execute_with_fallback(
            operation_name="LLM generation and CoT extraction",
            operation_func=helper_function,
            on_error=on_error,
            max_retries=max_retries,
            retry_delay=retry_delay,
            logger=logger,
            fallback_value=(None, None)
        )
        if error_msg and (on_error == "raise" or on_error == "retry"):
            # Log the error and raise an exception
            logger.error(f"LLM generation and CoT extraction failed: {error_msg}")
            raise RuntimeError(f"LLM generation and CoT extraction failed: {error_msg}")
        elif error_msg and on_error == "continue":
            # Log the error but continue
            logger.error(f"LLM generation and CoT extraction failed: {error_msg}")
            return None, None
        
        # If the operation was successful, unpack the result
        response, cot = result
        if response is None or cot is None:
            # Handle the case where the operation failed
            logger.error("LLM generation and CoT extraction returned None")
            return None, None
        # Return the generated response and parsed CoT
        return response, cot
    
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
        Verify a node and optionally update its status.
        
        Args:
            node: The node to verify
            question: Original question
            ground_truth_answer: The true answer
            verifier: Verification function to use
            on_error: How to handle verification errors:
                - "continue": Return false but don't raise an exception (skip this path)
                - "raise": Return error message to allow caller to handle
                - "retry": Retry the verification (useful for transient errors)
            max_retries: Maximum number of retry attempts if on_error="retry"
            retry_delay: Seconds to wait between retry attempts
                
        Returns:
            tuple[bool, str | None]: (verification_success, error_message if any)
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
            return False, None
        
        # Verification call was successful
        verification_result, explanation = result
        node.metadata["verification"] = explanation

        if verification_result:
            # Verification was successful
            node.success = True
            node.is_final = True

        return verification_result, explanation
            