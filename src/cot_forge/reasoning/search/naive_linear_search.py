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
from cot_forge.reasoning.strategies import Strategy, StrategyRegistry, default_strategy_registry
from cot_forge.reasoning.types import ReasoningNode, SearchResult
from cot_forge.reasoning.verifiers import default_verifier
from cot_forge.utils.parsing import extract_cot, extract_final_answer_from_cot

logger = logging.getLogger(__name__)

def random_strategy_selector(
    node: ReasoningNode,
    registry: StrategyRegistry
) -> Strategy:
    """Select a random strategy from the registry."""
    strategy_names = registry.list_strategies()
    
    # Filter out initial strategies if not the first step
    if node:
        strategy_names = [name for name in strategy_names 
                         if not registry.get_strategy(name).is_initial]
    else:
        # First step must be an initial strategy
        strategy_names = [name for name in strategy_names 
                         if registry.get_strategy(name).is_initial]
        
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
    verifier = default_verifier,
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
    
    # Get max_depth either from kwargs or default to 3
    max_depth = kwargs.get("max_depth", 3)
    
    current_node = None
    
    for depth in range(max_depth):
        # Select next strategy
        strategy = random_strategy_selector(current_node, strategy_registry)
        
        # Build prompt based on selected strategy
        current_cot = current_node.cot if current_node else None
        prompt = strategy.build_prompt(question, str(current_cot))
        
        # Generate response. Case where response generation fails, return failure response
        try:
            response = llm_provider.generate(
                prompt=prompt, 
                **(llm_kwargs or {})
            )
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return SearchResult(
                final_node=current_node,
                all_terminal_nodes=[current_node],
                success=False,
                final_answer=None,
                metadata={"error": str(e), "depth": depth}
            )
        
        # Case where parsing cot fails, return failure response
        try:
            cot = extract_cot(response)
        except Exception as e:
            logger.error(f"Error extracting CoT: {e}")
            return SearchResult(
                final_node=current_node,
                all_terminal_nodes=[current_node],
                success=False,
                final_answer=None,
                metadata={"error": str(e), "depth": depth}
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
        if verifier.verify(node=current_node,
                           question=question,
                           ground_truth_answer=ground_truth_answer,
                           llm_provider=llm_provider,
                           llm_kwargs=llm_kwargs or {}):
            return SearchResult(
                final_node=current_node,
                all_terminal_nodes=[current_node],
                success=True,
                final_answer=extract_final_answer_from_cot(current_node.cot),
                metadata={"depth": depth + 1, "reason": "verifier_success"}
            )
    
    # Max depth reached without success
    return SearchResult(
        final_node=current_node,
        all_terminal_nodes=[current_node],
        success=False,
        final_answer=extract_final_answer_from_cot(current_node.cot) if current_node else None,
        metadata={"depth": max_depth, "reason": "max_depth_reached"}
    )