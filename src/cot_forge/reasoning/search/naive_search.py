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
from cot_forge.reasoning.strategies import Strategy, StrategyRegistry, default_strategy_registry

from .search_algorithm import ReasoningStep, SearchResult, StrategySelector, TerminationChecker

logger = logging.getLogger(__name__)

def random_strategy_selector(
    question: str, 
    chain: list[ReasoningStep], 
    registry: StrategyRegistry
    
) -> Strategy:
    """Select a random strategy from the registry."""
    strategy_names = registry.list_strategies()
    
    # Filter out initial strategies if not the first step
    if chain:
        strategy_names = [name for name in strategy_names 
                         if not registry.get_strategy(name).is_initial]
    else:
        # First step must be an initial strategy
        strategy_names = [name for name in strategy_names 
                         if registry.get_strategy(name).is_initial]
    
    selected_name = random.choice(strategy_names)
    return registry.get_strategy(selected_name)

def is_correct_answer(chain: list[ReasoningStep], 
                      question: str,
                      ground_truth_answer: str,
                      llm_provider: LLMProvider,
                      llm_kwargs: dict[str, Any] | None) -> bool:
    """Default termination checker that looks for 'correct' in verification step."""
    if not chain:
        return False
    
    final_answer = extract_final_answer(chain[-1]["response"])
    
    if not final_answer:
        return False
    
    # LLM as judge verifier checks if the final answer is correct
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
        
        return response.strip().lower() == "yes"
    
    except Exception as e:
        logger.error(f"Error generating verification response: {e}")
        return False

def naive_search(
    question: str,
    ground_truth_answer: str,
    llm_provider: LLMProvider,
    termination_checker: TerminationChecker = is_correct_answer,
    strategy_selector: StrategySelector = random_strategy_selector,
    strategy_registry: StrategyRegistry = default_strategy_registry,
    llm_kwargs: dict[str, Any] | None = None,
    **kwargs
) -> SearchResult:
    """
    Perform a sequential search to generate a chain of thought.
    
    Args:
        question: The question to answer.
        ground_truth_answer: The true answer to the question.
        llm_provider: The LLM provider to use.
        max_depth: Maximum depth of the search.
        termination_checker: Function to check if search should terminate.
        strategy_registry: Registry of available strategies.
        strategy_selector: Function to select the next strategy.
        llm_kwargs: Additional kwargs for LLM provider.
        **kwargs: Additional kwargs for search algorithm.
    
    Returns:
        A SearchResult containing the chain of thought and metadata.
    """
    
    # Get max_depth either from kwargs or default to 3
    max_depth = kwargs.get("max_depth", 3)
    
    chain: list[ReasoningStep] = []
    
    for depth in range(max_depth):
        # Select next strategy
        strategy = strategy_selector(question, chain, strategy_registry)
        
        # Build prompt based on selected strategy
        previous_cot = chain[-1]["response"] if chain else None
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
                chain=chain,
                success=False,
                final_answer=None,
                metadata={"error": str(e), "depth": depth}
            )
        
        # Add step to chain
        step = ReasoningStep(
            strategy=strategy,
            prompt=prompt,
            response=response,
            is_final=False
        )
        chain.append(step)
        
        # Check for termination
        if termination_checker(chain, question):
            chain[-1]["is_final"] = True
            return SearchResult(
                chain=chain,
                success=True,
                final_answer=extract_final_answer(chain[-1]["response"]),
                metadata={"depth": depth + 1}
            )
    
    # Max depth reached without success
    return SearchResult(
        chain=chain,
        success=False,
        final_answer=extract_final_answer(chain[-1]["response"]) if chain else None,
        metadata={"depth": max_depth, "reason": "max_depth_reached"}
    )
    
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