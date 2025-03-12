"""
Beam search implementation for reasoning chains.
This search algorithm expands the reasoning chain by maintaining multiple paths
and selecting the most promising ones based on a scoring mechanism.
Branching factor options are chosen at random.

Logical flow:
1. Initialize the chain with an initial cot.
2. Randomly select beam_width x branching_factor strategies from the registry.
3. Use a scoring mechanism to select the most promising paths.
4. Expand each beam by choosing the best strategy for the current node.
5. Continue until a termination condition is met or max depth is reached.
6. Return the best chain(s) found.
"""
#TODO: Experiment with multithreading in beam generation
#TODO: Checker for at what depth different strategies are available


import logging
import random
from typing import Any

from cot_forge.llm import LLMProvider
from cot_forge.reasoning.strategies import (InitializeCoT, Strategy,
                                            StrategyRegistry,
                                            default_strategy_registry)
from cot_forge.reasoning.types import ReasoningNode, SearchResult
from cot_forge.reasoning.verifiers import BaseVerifier, default_verifier
from cot_forge.utils.parsing import extract_cot, extract_final_answer_from_cot
from cot_forge.utils.search_utils import (create_error_handler,
                                          intialize_blank_node)

logger = logging.getLogger(__name__)

def get_strategy_options(
    registry: StrategyRegistry,
    branching_factor: int = None,
    **kwargs: Any,
) -> list[Strategy]:
    """
    Selects a set of strategies options for a beam to consider.
    
    Args:
        registry: The strategy registry.
        branching_factor: Number of strategies to sample. If None, all strategies are considered.
        **kwargs: Additional arguments for strategy selection.
        
    Returns:
        A list of strategies to consider.
    """
    
    strategy_names = [strat for strat in registry.list_strategies() if not registry.get_strategy(strat).is_initial]
    
    if not strategy_names:
        raise ValueError("No appropriate strategies found for step")
    
    if branching_factor > len(strategy_names):
        raise ValueError("branching_factor cannot exceed number of valid strategies in registry")
    
    # Consider all strategies if no branching factor
    if branching_factor is None:
        branching_factor = len(strategy_names)
        
    return random.sample(strategy_names, branching_factor)

def intitialize_cot(
    question: str,
    llm_provider: LLMProvider,
    llm_kwargs: dict[str, Any] = None
) -> ReasoningNode:
    """
    Returns a reasoning node with the intitial CoT question and response.

    Args:
        question : The question
        llm_provider: The LLM provider to generate initial CoT

    Returns:
        ReasoningNode: Initial CoT reasoning node
    """
    strategy = InitializeCoT
    prompt = strategy.build_prompt(question)
    
    current_node = ReasoningNode(strategy=strategy,
                                 prompt=prompt,
                                 response="",
                                 cot=None,
                                 parent=None)
    
    # Generate response, return None node if fails
    try:
        response = llm_provider.generate(
            prompt = prompt,
            **(llm_kwargs or {})
        )
    except Exception as e:
        return handle_search_error(error=e,
                            current_node=current_node,
                            logger=logger,
                            metadata={"beams": 0, "depth": 0})
    current_node.response = response
        
    try:
        cot = extract_cot(response)
    except Exception as e:
        return handle_search_error(error=e,
                            current_node=current_node,
                            logger=logger,
                            metadata={"beams": 0, "depth": 0})
    current_node.cot = cot
    
    return current_node

def generate_cot(node: ReasoningNode,
                 strategy: Strategy,
                 llm_provider: LLMProvider,
                 llm_kwargs: dict[str, Any] = None
                 ) -> tuple['str', dict['str', 'str']]:
    """
    Generate a chain of thought using the provided strategy.
    Args:
        node: Current deepest node in the beam.
        strategy: Strategy to use for generating the next step.
        llm_provider: LLM provider to generate the response.
    Returns:
        (str, dict): Response and  dictionary containing the generated chain of thought.
    """
    prompt = strategy.build_prompt(question=node.prompt, previous_cot=node.cot)
    
    # Generate response.
    try: 
        response = llm_provider.generate(
            prompt=prompt,
            **(llm_kwargs or {})
        )
        # Extract CoT from response
        cot = extract_cot(response)
        return response, cot
    
    except Exception as e:
        logger.error(f"Error in generating response: {e}")
        return None, None


def random_sample_beam_search(
    question: str,
    ground_truth_answer: str,
    llm_provider: LLMProvider,
    verifier: BaseVerifier = default_verifier,
    scorer = likert_scorer,
    strategy_registry: StrategyRegistry = default_strategy_registry,
    llm_kwargs: dict[str, Any] = None,
    **kwargs
) -> SearchResult:
    """
    Perform a beam search to generate possible chains of thought.
    
    Args:
        question: The question to answer.
        ground_truth_answer: The true answer to the question.
        llm_provider: The LLM provider to use.
        verifier: The verifier to use for checking correctness of final answers.
        scorer: The scorer used to evaluate different beam options.
        strategy_registry: Registry of available strategies.
        llm_kwargs: Additional kwargs for LLM provider.
        **kwargs: Additional kwargs for search algorithm.
        
    Returns:
        A SearchResult containing terminal_nodes.
    """
    
    # TODO: Consider how to get these out of kwargs and into parameters, including in naive linear
    # Get max_depth either from kwargs or default to 3
    max_depth = kwargs.get("max_depth", 3)
    
    # Try to get beam_width from kwargs, default to 3
    beam_width = kwargs.get("beam_width", 3)
    
    # Try to get branching_factor from kwargs, default to 3
    branching_factor = kwargs.get("branching_factor", 3)
    
    # Track terminal nodes for failure cases
    terminal_nodes = []
    
    # Create initial node
    current_node = intitialize_cot(question=question,
                                   llm_provider=llm_provider,
                                   llm_kwargs=llm_kwargs)
    
    # Create beams with blank nodes and initial node as parent
    beams = [intialize_blank_node(parent=current_node) for _ in range(beam_width)]
    
    for depth in range(max_depth):
        for beam in beams:
            strategy_options = get_strategy_options(registry=strategy_registry,branching_factor=branching_factor)
            
            # Generate CoT for each strategy option
            cot_options = []
            for strat_name in strategy_options:
                strategy = strategy_registry.get_strategy(strat_name)
                response, cot = generate_cot(
                    node=beam,
                    strategy=strategy,
                    llm_provider=llm_provider,
                    llm_kwargs=llm_kwargs
                )
                # If response or cot is None, indicates an error. Skip this option.
                if response is None or cot is None:
                    continue
                cot_options.append((response, cot))
                
            # Now we have a list of cot options for this beam
            # Score each option using the provided scorer
            
