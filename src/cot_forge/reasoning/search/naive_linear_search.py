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
from cot_forge.reasoning.strategies import (Strategy, StrategyRegistry,
                                            default_strategy_registry)
from cot_forge.reasoning.types import ReasoningNode, SearchResult
from cot_forge.reasoning.verifiers import BaseVerifier
from cot_forge.utils.parsing import extract_cot, extract_final_answer_from_cot
from cot_forge.utils.search_utils import try_operation

from .search_algorithm import BaseSearch

logger = logging.getLogger(__name__)

class NaiveLinearSearch(BaseSearch):
    """
    Naive linear search for reasoning chain.
    
    This class implements a naive sequential search algorithm to generate a chain of thought (CoT).
    It selects strategies randomly from the registry and continues until the verifier returns true
    or the maximum depth is reached.
    
    Attributes:
        max_depth (int): Maximum depth for the search.
    """
    
    def __init__(self,
                 max_depth: int = 3,
                 strategy_registry: StrategyRegistry = default_strategy_registry,
                 ):
        super().__init__()
        self.strategy_registry = strategy_registry
        self.max_depth = max_depth
        self.name = "naive linear search"
        self.description = "Naive linear search for reasoning chain."

    def random_strategy_selector(
        self,
        node: ReasoningNode,
        registry: StrategyRegistry,
        depth: int = 0
    ) -> Strategy:
        """Select a random strategy from the registry.
        
        Args:
            node: The current reasoning node.
            registry: The strategy registry.
            depth: The current depth in the reasoning chain.
            
        Returns:
            A randomly selected strategy.
        """
        strategy_names = registry.list_strategies()
        
        # Filter out initial strategies if not the first step
        if depth == 0:
            # First step must be the initial strategy
            strategy = registry.get_strategy("initialize")
            return strategy
        else:
            strategy_names = [
                name for name in strategy_names 
                if not registry.get_strategy(name).is_initial
                and depth >= registry.get_strategy(name).minimum_depth
            ]
            
        if not strategy_names:
            step_type = "initial" if node is None else "continuation"
            raise ValueError(f"No appropriate strategies found for {step_type} step")

        selected_name = random.choice(strategy_names)
        logger.debug(f"Selected strategy: {selected_name}")
        
        return registry.get_strategy(selected_name)

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
            strategy = self.random_strategy_selector(current_node, self.strategy_registry, depth)
            
            # Build prompt based on selected strategy
            full_cot = current_node.get_full_cot() if current_node else None
            prompt = strategy.build_prompt(question, str(full_cot))
            
            # Generate response. Case where response generation fails, return failure response
            response, error = try_operation(
                "LLM generation",
                reasoning_llm.generate,
                kwargs={"node": current_node, "prompt": prompt, 'llm_kwargs': llm_kwargs},
                error_action="return",
                logger=logger
            )

            if error:
                return SearchResult(
                    final_node=current_node,
                    all_terminal_nodes=[current_node],
                    success=False,
                    final_answer=None,
                    metadata={"depth": depth,
                              "reason": "llm_generation_error",
                              "question": question,
                              "ground_truth_answer": ground_truth_answer}
                )
                
            # Extract CoT from response with error handling
            cot, error = try_operation(
                "CoT extraction",
                extract_cot,
                kwargs={"response": response},
                error_action="return",
                logger=logger
            )
            if error:
                return SearchResult(
                    final_node=current_node,
                    all_terminal_nodes=[current_node],
                    success=False,
                    final_answer=None,
                    metadata={"depth": depth,
                              "reason": "cot_extraction_error",
                              "question": question,
                              "ground_truth_answer": ground_truth_answer}
                )
            
            # Create new reasoning node and incorporate into graph
            previous_node = current_node if current_node else None    
            current_node = ReasoningNode(
                strategy=strategy,
                prompt=prompt,
                response=response,
                cot = cot,
                parent=previous_node
            )
            
            if previous_node:
                previous_node.add_child(current_node)
            
            # Check for success condition by verifier
            verification_result, explanation = verifier(
                node=current_node,
                question=question,
                ground_truth_answer=ground_truth_answer,
            )
                    
            # Append the verification explanation to the node's cot
            current_node.metadata.update({"verification": explanation})
            
            # DEBUG
            # verification_result = False
            
            # If verification is successful, return the result
            if verification_result:
                current_node.is_final = True
                current_node.success = True
                return SearchResult(
                    final_node=current_node,
                    all_terminal_nodes=[current_node],
                    success=True,
                    final_answer=extract_final_answer_from_cot(current_node.cot),
                    metadata={"depth": depth + 1,
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
                      "reason": "max_depth_reached",
                      "question": question,
                      "ground_truth_answer": ground_truth_answer}
        )