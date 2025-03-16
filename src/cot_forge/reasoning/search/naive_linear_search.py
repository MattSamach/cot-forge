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
from cot_forge.reasoning.verifiers import BaseVerifier, default_verifier
from cot_forge.utils.parsing import extract_cot, extract_final_answer_from_cot
from cot_forge.utils.search_utils import create_error_handler, try_operation

logger = logging.getLogger(__name__)

def random_strategy_selector(
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

def naive_linear_search(
    question: str,
    ground_truth_answer: str,
    llm_provider: LLMProvider,
    verifier: BaseVerifier = default_verifier,
    strategy_registry: StrategyRegistry = default_strategy_registry,
    llm_kwargs: dict[str, Any] = None,
    **kwargs
) -> SearchResult:
    """
    Perform a naive/random sequential search to generate a chain of thought.
    
    Args:
        question: The question to answer.
        ground_truth_answer: The true answer to the question.
        llm_provider: The LLM provider to use.
        verifier: The verifier to use for checking correctness.
        strategy_registry: Registry of available strategies.
        llm_kwargs: Additional kwargs for LLM provider.
        **kwargs: Additional kwargs for search algorithm.
    
    Returns:
        A SearchResult containing the chain of thought and metadata.
    """
    
    # Helper functions to manage errors
    handle_error = create_error_handler(logger)

    # TODO: Figure out how to get out of kwargs into function parameters
    # Get max_depth either from kwargs or default to 3
    max_depth = kwargs.get("max_depth", 3)
    
    current_node = None
    
    for depth in range(max_depth):
        # Select next strategy
        strategy = random_strategy_selector(current_node, strategy_registry, depth)
        
        # Build prompt based on selected strategy
        full_cot = current_node.get_full_cot() if current_node else None
        prompt = strategy.build_prompt(question, str(full_cot))
        
        # Generate response. Case where response generation fails, return failure response
        response, error = try_operation(
            "LLM generation",
            llm_provider.generate,
            kwargs={"node": current_node, "prompt": prompt, 'llm_kwargs': (llm_kwargs or {})},
            error_action="return",
            logger=logger
        )
        # response = handle_error(func=llm_provider.generate,
        #                         node=current_node,
        #                         metadata={"depth": depth},
        #                         prompt=prompt,
        #                         llm_kwargs=llm_kwargs)
        if error:
            return SearchResult(
                final_node=current_node,
                all_terminal_nodes=[current_node],
                success=False,
                final_answer=None,
                metadata={"depth": depth, "reason": "llm_generation_error", "question": question, "ground_truth_answer": ground_truth_answer}
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
                metadata={"depth": depth, "reason": "cot_extraction_error", "question": question, "ground_truth_answer": ground_truth_answer}
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
            llm_provider=llm_provider,
            llm_kwargs=llm_kwargs or {}
        )
                
        # Append the verification explanation to the node's cot
        current_node.metadata.update({"verification": explanation})
        
        # If verification is successful, return the result
        if verification_result:
            current_node.is_final = True
            current_node.success = True
            return SearchResult(
                final_node=current_node,
                all_terminal_nodes=[current_node],
                success=True,
                final_answer=extract_final_answer_from_cot(current_node.cot),
                metadata={"depth": depth + 1, "reason": "verifier_success", "question": question, "ground_truth_answer": ground_truth_answer},
            )
    
    # Max depth reached without success
    return SearchResult(
        final_node=current_node,
        all_terminal_nodes=[current_node],
        success=False,
        final_answer=extract_final_answer_from_cot(current_node.cot) if current_node else None,
        metadata={"depth": max_depth, "reason": "max_depth_reached", "question": question, "ground_truth_answer": ground_truth_answer}
    )