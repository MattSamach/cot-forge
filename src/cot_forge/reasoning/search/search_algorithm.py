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

class ReasoningNode:
    """A node in the reasoning graph/tree/chain."""
    def __init__(self, 
                 strategy: Strategy,
                 prompt: str,
                 response: str,
                 parent: Optional['ReasoningNode'] = None):
        self.strategy = strategy
        self.prompt = prompt
        self.response = response
        self.parent = parent
        self.children: list['ReasoningNode'] = []
        self.is_final = False
        self.metadata: dict[str, Any] = {}

    def add_child(self, child: 'ReasoningNode'):
        self.children.append(child)
        
    def get_full_chain(self):
        """Get the complete chain from the root to this node."""
        chain = []
        current_node = self
        while current_node:
            chain.append(current_node)
            current_node = current_node.parent
        return list(reversed(chain))

class SearchResult(TypedDict):
    """Represents the result of a search algorithm."""
    final_node: ReasoningNode
    all_terminal_nodes: list[ReasoningNode] | None
    success: bool
    final_answer: Optional[str]
    metadata: dict[str, Any]

class TerminationChecker(Protocol):
    """Protocol for termination checking functions."""
    def __call__(self, 
                node: ReasoningNode,
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
