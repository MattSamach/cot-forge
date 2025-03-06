"""
Naïve sequential search for reasoning chains.

Logical flow:
1. Initialize chain with Initialize strategy.
2. Randomly select a strategy from the registry.
3. Continue until termination condition is met or max depth is reached.
"""
import json
import logging
import random
from typing import Any

from cot_forge.llm import LLMProvider
from cot_forge.reasoning.strategies import (Strategy, StrategyRegistry,
                                            default_strategy_registry)

from .search_algorithm import ReasoningNode, SearchResult, TerminationChecker

logger = logging.getLogger(__name__)

def random_strategy_selector(
    question: str, 
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
        raise ValueError(f"No appropriate strategies found for {'initial' if node is None else 'continuation'} step")

    selected_name = random.choice(strategy_names)
    logger.debug(f"Selected strategy: {selected_name}")
    
    return registry.get_strategy(selected_name)

# TODO: Move to verifiers folder
def is_correct_answer(node: ReasoningNode, 
                      question: str,
                      ground_truth_answer: str,
                      llm_provider: LLMProvider,
                      llm_kwargs: dict[str, Any] | None) -> bool:
    """Default termination checker that looks for 'correct' in verification step."""
    
    if not node:
        return False
    
    final_answer = extract_final_answer(node.response)
    
    if not final_answer:
        return False
    
    # LLM as judge verifier checks if the final answer is correct
    # TODO: Move to verifiers.prompts.py file
    try:
        response = llm_provider.generate(
            prompt=f"""You are an answer judge.
            Verify if the provided answer below is equivalent to the ground truth answer.
            Answer simply with "yes" or "no".
            
            Provided answer: {final_answer}
            Ground truth answer: {ground_truth_answer}
            """,
            **llm_kwargs
        )
        response = response.strip().lower()
        
        if not response:
            raise ValueError("Empty response from LLM")
        
        return response.strip().lower() == "yes"
    
    except Exception as e:
        logger.error(f"Error generating verification response: {e}")
        return False

def naive_linear_search(
    question: str,
    ground_truth_answer: str,
    llm_provider: LLMProvider,
    termination_checker: TerminationChecker = is_correct_answer,
    strategy_registry: StrategyRegistry = default_strategy_registry,
    llm_kwargs: dict[str, Any] | None = None,
    **kwargs
) -> SearchResult:
    """
    Perform a naive/random sequential search to generate a chain of thought.
    
    Args:
        question: The question to answer.
        ground_truth_answer: The true answer to the question.
        llm_provider: The LLM provider to use.
        termination_checker: Function to check if search should terminate.
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
        strategy = random_strategy_selector(question, current_node, strategy_registry)
        
        # Build prompt based on selected strategy
        previous_cot = current_node.parent.response if current_node and current_node.parent else None
        prompt = strategy.build_prompt(question, previous_cot)
        
        # Generate response
        try:
            response = llm_provider.generate(
                prompt=prompt, 
                **llm_kwargs
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
        
        # Create new reasoning node and incorporate into graph
        previous_node = current_node if current_node else None
        current_node = ReasoningNode(
            strategy=strategy,
            prompt=prompt,
            response=response,
            parent=previous_node
        )
        
        if previous_node:
            previous_node.add_child(current_node)
        
        # Check for termination
        if termination_checker(node=current_node,
                               question=question,
                               ground_truth_answer=ground_truth_answer,
                               llm_provider=llm_provider,
                               llm_kwargs=llm_kwargs):
            current_node.is_final = True
            return SearchResult(
                final_node=current_node,
                all_terminal_nodes=[current_node],
                success=True,
                final_answer=extract_final_answer(current_node.response),
                metadata={"depth": depth + 1, "reason": "termination_checker"}
            )
    
    # Max depth reached without success
    return SearchResult(
        final_node=current_node,
        all_terminal_nodes=[current_node],
        success=False,
        final_answer=extract_final_answer(current_node.response) if current_node else None,
        metadata={"depth": max_depth, "reason": "max_depth_reached"}
    )
    
# TODO: Move to utils
def extract_final_answer(response: str) -> str:
    """Extract the final answer from a response."""
    try:
        data = json.loads(response)
        
        for action in reversed(data.get("CoT", [])):
            if action.get("action") == "Final Conclusion":
                return action.get("content", "")
            
    except json.JSONDecodeError as err:
        raise ValueError("Invalid JSON response") from err

    return "No final answer found"