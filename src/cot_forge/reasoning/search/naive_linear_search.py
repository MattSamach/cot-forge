"""
Naïve sequential search for reasoning chains.

Logical flow:
1. Initialize chain with Initialize strategy.
2. Randomly select a strategy from the registry.
3. Continue until verifier returns true or max depth is reached.
"""

import logging
import random
from typing import Any

from cot_forge.llm import LLMProvider
from cot_forge.reasoning.strategies import (RandomStrategySelector,
                                            StrategyRegistry,
                                            default_strategy_registry)
from cot_forge.reasoning.types import SearchResult
from cot_forge.reasoning.verifiers import BaseVerifier
from cot_forge.utils.parsing import extract_final_answer_from_cot
from cot_forge.utils.search_utils import generate_and_parse_cot

from .search_algorithm import BaseSearch

logger = logging.getLogger(__name__)

class NaiveLinearSearch(BaseSearch):
    """
    Naive linear search for reasoning chain.
    
    This class implements a naive sequential search algorithm to generate a chain of thought (CoT).
    It selects strategies randomly from the registry and continues until the verifier returns true
    or the maximum depth is reached.
    
    Attributes:
        strategy_registry (StrategyRegistry): The strategy registry to use for selecting strategies.
        max_depth (int): Maximum depth for the search.
        name (str): Name of the search algorithm.
        description (str): Description of the search algorithm.
    """
    
    def __init__(self,
                 max_depth: int = 3,
                 strategy_registry: StrategyRegistry = default_strategy_registry,
                 ):
        self.strategy_registry = strategy_registry
        self.max_depth = max_depth
        self.name = "naive linear search"
        self.description = "Naive linear search for reasoning chain."
        self.strategy_selector = RandomStrategySelector()

    def _search(
        self,
        question: str,
        ground_truth_answer: str,
        verifier: BaseVerifier,
        reasoning_llm: LLMProvider,
        llm_kwargs: dict[str, Any] = None,
        **kwargs
    ) -> SearchResult:
        """
        Perform a naive/random sequential search to generate a chain of thought.
        
        Args:
            question: The question to answer.
            ground_truth_answer: The true answer to the question.
            verifier: The verifier to use for checking correctness.
            reasoning_llm: The LLM provider to use.
            llm_kwargs: Additional kwargs for reasoning LLM calls.
            **kwargs: Additional kwargs for search algorithm.
        
        Returns:
            A SearchResult containing the chain of thought and metadata.
        """
                
        llm_kwargs = llm_kwargs or {}
        
        current_node = None
        
        for depth in range(self.max_depth):
            # Select next strategy
            strategies, _ = self.strategy_selector.select(registry = self.strategy_registry, depth = depth)
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
            except:
                return SearchResult(
                    final_node=current_node,
                    all_terminal_nodes=[current_node] if current_node else [],
                    success=False,
                    final_answer=None,
                    metadata={"depth": depth,
                              "reason": "generation_error",
                              "question": question,
                              "ground_truth_answer": ground_truth_answer}
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
                    final_node=current_node,
                    all_terminal_nodes=[current_node],
                    success=True,
                    final_answer=extract_final_answer_from_cot(current_node.cot),
                    metadata={"depth": depth + 1,
                              "max_depth": self.max_depth,
                              "reason": "verifier_success",
                              "question": question,
                              "ground_truth_answer": ground_truth_answer},
                )
        
        # Max depth reached without success
        return SearchResult(
            final_node=current_node,
            all_terminal_nodes=[current_node],
            success=False,
            final_answer=extract_final_answer_from_cot(current_node.cot) if current_node else None,
            metadata={"depth": self.max_depth,
                      "max_depth": self.max_depth,
                      "reason": "max_depth_reached",
                      "question": question,
                      "ground_truth_answer": ground_truth_answer}
        )