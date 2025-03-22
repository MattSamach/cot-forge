"""
Implements a naive linear search for reasoning chains.

This search algorithm explores the reasoning space by sequentially applying
randomly selected strategies from a predefined registry. It starts with an
initial state (usually defined by an "Initialize" strategy) and iteratively
expands the chain of thought (CoT) until a verification condition is met or
a maximum depth is reached.

The logical flow of the search is as follows:
1.  Initialize the chain of thought (CoT) using an initial strategy.
2.  Randomly select a strategy from the strategy registry.
3.  Apply the selected strategy to extend the CoT.
4.  Verify the extended CoT against a ground truth answer using a verifier.
5.  Repeat steps 2-4 until the verifier returns True (success) or the
    maximum search depth is reached (failure).
"""

import logging
from typing import Any

from cot_forge.llm import LLMProvider
from cot_forge.reasoning.strategies import RandomStrategySelector, StrategyRegistry, default_strategy_registry
from cot_forge.reasoning.types import SearchResult
from cot_forge.reasoning.verifiers import BaseVerifier
from cot_forge.utils.search_utils import generate_and_parse_cot

from .search_algorithm import BaseSearch

logger = logging.getLogger(__name__)

class NaiveLinearSearch(BaseSearch):
    """
    Naive linear search for reasoning chain.
    
    This class implements a naive sequential search algorithm to generate a chain of thought (CoT).
    It selects strategies randomly from the registry and continues until the verifier returns True
    or the maximum depth is reached.
    
    Attributes:
        strategy_registry (StrategyRegistry): The strategy registry to use for selecting strategies.
        max_depth (int): Maximum depth for the search.
        name (str): Name of the search algorithm.
        description (str): Description of the search algorithm.
        strategy_selector (StrategySelector): Strategy selector for choosing strategies.
    """
    
    def __init__(self,
                 max_depth: int = 3):
        self.max_depth = max_depth
        self.name = "Naive Linear Search"
        self.description = ("A sequential search algorithm that randomly selects " 
                            "and applies reasoning strategies to build a chain of thought. "
                            "Continues until verification succeeds or max depth is reached.")
        self.strategy_selector = RandomStrategySelector()
        
    def to_dict(self):
        """
        Convert the search algorithm to a dictionary representation.

        Returns:
            dict: Dictionary representation of the search algorithm.
        """
        return {
            "name": self.name,
            "description": self.description,
            "max_depth": self.max_depth,
        }
        
    @classmethod
    def from_dict(cls, data: dict):
        """
        Create a search algorithm instance from a dictionary representation.

        Args:
            data (dict): Dictionary representation of the search algorithm.

        Returns:
            NaiveLinearSearch: Instance of the search algorithm.
        """
        return cls(
            max_depth=data.get("max_depth", 3),
        )

    def _search(
        self,
        question: str,
        ground_truth_answer: str,
        verifier: BaseVerifier,
        reasoning_llm: LLMProvider,
        strategy_registry: StrategyRegistry = default_strategy_registry,
        llm_kwargs: dict[str, Any] = None,
        **kwargs
    ) -> SearchResult:
        """
        Perform a naive sequential search to generate a chain of thought (CoT).

        This method iteratively applies reasoning strategies to expand the CoT. At each step, it verifies
        the generated CoT against the ground truth answer using the provided verifier. The process continues
        until the verifier returns True or the maximum search depth is reached.

        Args:
            question (str): The question to answer.
            ground_truth_answer (str): The true answer to the question.
            verifier (BaseVerifier): The verifier used to check the correctness of the CoT.
            reasoning_llm (LLMProvider): The LLM provider used to generate reasoning steps.
            strategy_registry (StrategyRegistry): The strategy registry to use for selecting strategies.
            llm_kwargs (dict[str, Any], optional): Additional keyword arguments for the LLM provider.
            **kwargs: Additional keyword arguments for the search algorithm.

        Returns:
            SearchResult: Object with final reasoning node, success status, final answer, and metadata.

        Raises:
            Exception: If an error occurs during LLM generation or verification.

        Example:
            ```python
            search = NaiveLinearSearch(max_depth=5)
            result = search._search(
                question="What is the capital of France?",
                ground_truth_answer="Paris",
                verifier=my_verifier,
                reasoning_llm=my_llm_provider
            )
            print(result.success)  # True or False
            ```
        """
                
        llm_kwargs = llm_kwargs or {}
        
        # Initialize the reasoning node
        current_node = None
        
        for depth in range(self.max_depth):
            # Select next strategy
            strategies, _ = self.strategy_selector.select(registry = strategy_registry, depth = depth)
            strategy = strategies[0] if isinstance(strategies, list) else strategies
            
            # Build prompt based on selected strategy
            full_cot = current_node.get_full_cot() if current_node else None
            prompt = strategy.build_prompt(question, str(full_cot))
            
            # Generate response and cot.
            try:
                response, cot = generate_and_parse_cot(
                    reasoning_llm=reasoning_llm,
                    prompt=prompt,
                    llm_kwargs=llm_kwargs,
                    logger=logger
                )
            except Exception as e:
                logger.error(f"Error during LLM generation: {e}")
                return SearchResult(
                    question=question,
                    ground_truth_answer=ground_truth_answer,
                    terminal_nodes=[current_node] if current_node else [],
                    success=False,
                    metadata={"depth": depth,
                              "reason": "generation_error"}
                )
            
            # Create new reasoning node and incorporate into graph
            previous_node = current_node if current_node else None    
            current_node = self.create_node(
                strategy=strategy,
                prompt=prompt,
                response=response,
                cot=cot,
                parent=previous_node
            )
            
            # Check for success condition by verifier
            verification_result, explanation = self.verify_node(
                node=current_node,
                question=question,
                ground_truth_answer=ground_truth_answer,
                verifier=verifier,
                on_error="retry",
                max_retries=3,
                logger=logger
            )
            logger.info(f"Verification result: {verification_result}, Explanation: {explanation}")
                        
            # If verification is successful, return the result
            if verification_result:
                return SearchResult(
                    question=question,
                    ground_truth_answer=ground_truth_answer,
                    terminal_nodes=[current_node],
                    success=True,
                    metadata={"depth": depth + 1,
                              "reason": "verifier_success"},
                )
        
        # Max depth reached without success
        return SearchResult(
            question=question,
            ground_truth_answer=ground_truth_answer,
            terminal_nodes=[current_node],
            success=False,
            metadata={"depth": self.max_depth,
                      "reason": "max_depth_reached"}
        )