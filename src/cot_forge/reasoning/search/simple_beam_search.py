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

import logging
import random
from collections import defaultdict
from typing import Any

from cot_forge.llm import LLMProvider
from cot_forge.reasoning.scorers import BaseScorer
from cot_forge.reasoning.strategies import (InitializeCoT, Strategy,
                                            StrategyRegistry,
                                            default_strategy_registry)
from cot_forge.reasoning.types import ReasoningNode, SearchResult
from cot_forge.reasoning.verifiers import BaseVerifier, default_verifier
from cot_forge.utils.parsing import extract_cot

logger = logging.getLogger(__name__)

def get_strategy_options(
    registry: StrategyRegistry,
    branching_factor: int = None,
    depth: int = None,
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
    
    strategy_names = registry.list_strategies()
    
    strategy_names = [
        name for name in strategy_names 
        if not registry.get_strategy(name).is_initial
        and depth >= registry.get_strategy(name).minimum_depth
    ]
    
    if not strategy_names:
        raise ValueError("No appropriate strategies found for step")
    
    if branching_factor > len(strategy_names):
        raise ValueError("branching_factor cannot exceed number of valid strategies in registry")
    
    # Consider all strategies if no branching factor
    if branching_factor is None:
        branching_factor = len(strategy_names)
        
    return random.sample(strategy_names, branching_factor)

def initialize_cot(
    question: str,
    llm_provider: LLMProvider,
    llm_kwargs: dict[str, Any] = None
) -> ReasoningNode :
    """
    Returns a reasoning node with the initial CoT question and response.

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
        logger.error(f"Error in generating initial response: {e}")
        return None
        
    current_node.response = response
        
    try:
        cot = extract_cot(response)
    except Exception as e:
        return None
    
    current_node.cot = cot
    
    return current_node


def generate_cot(node: ReasoningNode,
                 strategy: Strategy,
                 llm_provider: LLMProvider,
                 llm_kwargs: dict[str, Any] = None
                 ) -> tuple['str', dict['str', 'str'], 'str']:
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
        return response, cot, prompt
    
    except Exception as e:
        logger.error(f"Error in generating response: {e}")
        return None, None, prompt
    
def initialize_beams(initial_node: ReasoningNode,
                     strategy_registry: StrategyRegistry,
                     beam_width: int,
                     scorer: BaseScorer,
                     branching_factor: int,
                     depth: int,
                     llm_provider: LLMProvider,
                     question: str,
                     ground_truth_answer: str,
                     verifier: BaseVerifier,
                     llm_kwargs: dict[str, Any] = {},
                     )-> list[ReasoningNode]:
    """
    Initialize the beams for the beam search by creating [beam_width] nodes. Each initialized beam 
    should try to employ different strategies to explore the search space.
    Args:
        node: Current node to initialize beams from.
        beam_width: Number of beams to create.
        scorer: Scorer to evaluate the beams.
        branching_factor: Number of strategies to consider for each beam.
    """
    
    strategy_options = get_strategy_options(registry = strategy_registry,
                                            branching_factor = branching_factor,
                                            depth = depth)
    
    # Generate CoT for each strategy option
    strategies_tracker = defaultdict(dict)
    for strat_name in strategy_options:
        strategy = strategy_registry.get_strategy(strat_name)
        response, cot, prompt = generate_cot(
            node=initial_node,
            strategy=strategy,
            llm_provider=llm_provider,
            llm_kwargs=llm_kwargs
        )
        # If response or cot is None, indicates an error. Skip this option.
        if response is None or cot is None:
            continue
        # Store the strategy, response, and cot in the strategies tracker
        strategies_tracker[strat_name]['response'] = response
        strategies_tracker[strat_name]['cot'] = cot
        strategies_tracker[strat_name]['prompt'] = prompt
        strategies_tracker[strat_name]['strategy'] = strategy
        
    # Score all options at once
    if not strategies_tracker:
        logger.error("No valid strategies found")
        return []

    # Create cot_list in the format expected by the scorer
    cot_list = [
            {"strategy_name": strat_name, "cot": strat_data['cot']}
            for strat_name, strat_data in strategies_tracker.items()
        ]
                
    # Score each option using the provided scorer
    # scores should be dictionary of form {"strategy_name": score...}
    scores = scorer(
        cot_list=cot_list,
        question=question,
        ground_truth_answer=ground_truth_answer,
        llm_provider=llm_provider,
        llm_kwargs=llm_kwargs
    )
    
    # Update scores
    for name, score in scores.items():
        strategies_tracker[name]['score'] = score
            
    # Select the best strategy based on the scores.
    # Loops through and repeats in cases where the number of strategies is less than the beam width.
    # Continues until the beam width is reached.
    selected_strategies = []
    sorted_strategies = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    i = 0
    while len(selected_strategies) < beam_width:
        strategy_name, score = sorted_strategies[i]
        selected_strategies.append(strategy_name)
        i += 1
        if i >= len(sorted_strategies):
            i = 0
            
    # Create beams with blank nodes and initial node as parent
    beams = []
    for strategy_name in selected_strategies:
        strat_data = strategies_tracker[strategy_name]
        new_beam = ReasoningNode(
            strategy=strat_data['strategy'],
            prompt=strat_data['prompt'],
            response=strat_data['response'],
            cot=strat_data['cot'],
            parent=initial_node,
            metadata={"score": strat_data['score']}
        )
        initial_node.add_child(new_beam)
        beams.append(new_beam)
        
    # Check if the new node is a terminal node with verifier
    for beam in beams:
        verification_result, explanation = verifier(
            node=beam,
            question=question,
            ground_truth_answer=ground_truth_answer,
            llm_provider=llm_provider,
            llm_kwargs=llm_kwargs or {}
        )
            
        # Append the verification explanation to the node's cot
        beam.cot.append({"action": "verification", "content": explanation})

        # If verification is successful, mark the node as final
        # and add it to the terminal nodes
        if verification_result:
            beam.is_success = True
            beam.is_final = True
            continue
        
    return beams

def simple_beam_search(
    question: str,
    ground_truth_answer: str,
    llm_provider: LLMProvider,
    scorer: BaseScorer,
    verifier: BaseVerifier = default_verifier,
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
        scorer: The scorer used to evaluate different beam options.
        verifier: The verifier to use for checking correctness of final answers.
        strategy_registry: Registry of available strategies.
        llm_kwargs: Additional kwargs for LLM provider.
        **kwargs: Additional kwargs for search algorithm.
        
    Returns:
        A SearchResult containing terminal nodes of beams.
    """
    
    # TODO: Consider how to get these out of kwargs and into parameters, including in naive linear search
    # Get max_depth either from kwargs or default to 3
    max_depth = kwargs.get("max_depth", 3)
    
    # Try to get beam_width from kwargs, default to 3
    beam_width = kwargs.get("beam_width", 3)
    
    # Try to get branching_factor from kwargs, default to 3
    branching_factor = kwargs.get("branching_factor", 3)
    
    # Create initial node
    current_node = initialize_cot(question=question,
                                   llm_provider=llm_provider,
                                   llm_kwargs=llm_kwargs)
    
    if current_node is None:
        logger.error("Failed to initialize CoT. Exiting.")
        return SearchResult(all_terminal_nodes=None, 
                            question=question,
                            ground_truth_answer=ground_truth_answer,
                            success=False,
                            metadata={"error": "Failed to initialize CoT"})
        
    # Initialize the beams
    try:
        beams = initialize_beams(
            initial_node=current_node,
            strategy_registry=strategy_registry,
            llm_provider=llm_provider,
            llm_kwargs=llm_kwargs,
            question=question,
            ground_truth_answer=ground_truth_answer,
            beam_width=beam_width,
            scorer=scorer,
            verifier=verifier,
            branching_factor=branching_factor,
            depth=1
        )
    except Exception as e:
        logger.error(f"Error in initializing beams: {e}")
        return SearchResult(all_terminal_nodes=None, 
                            question=question,
                            ground_truth_answer=ground_truth_answer,
                            success=False,
                            metadata={"error": "Failed to initialize beams"})
        
    # Range starts at 2 because we already have the initial node and beams
    # We will expand the beams at each depth
    for depth in range(2, max_depth):
        # Check if all beams are final
        if all(beam.is_final for beam in beams):
            break
        
        # Expand each beam
        for i, beam in enumerate(beams):
            # If the beam is final, skip it
            if beam.is_final:
                continue
            
            strategy_options = get_strategy_options(registry = strategy_registry,
                                                    branching_factor = branching_factor,
                                                    depth = depth)
            
            # Generate CoT for each strategy option
            strategies_tracker = defaultdict(dict)
            for strat_name in strategy_options:
                strategy = strategy_registry.get_strategy(strat_name)
                response, cot, prompt = generate_cot(
                    node=beam,
                    strategy=strategy,
                    llm_provider=llm_provider,
                    llm_kwargs=llm_kwargs
                )
                # If response or cot is None, indicates an error. Skip this option.
                if response is None or cot is None:
                    continue
                # Store the strategy, response, and cot in the strategies tracker
                strategies_tracker[strat_name]['response'] = response
                strategies_tracker[strat_name]['cot'] = cot
                strategies_tracker[strat_name]['prompt'] = prompt
                strategies_tracker[strat_name]['strategy'] = strategy
                
            if not strategies_tracker:
                logger.error("No valid strategies found")
                raise ValueError("No valid strategies found")
            
            cot_list = [
                {"strategy_name": strat_name, "cot": strat_data['cot']}
                for strat_name, strat_data in strategies_tracker.items()
            ]
            
            # Score each option using the provided scorer
            # scores should be dictionary of form {"strategy_name": score...}
            scores = scorer(
                cot_list=cot_list,
                question=question,
                ground_truth_answer=ground_truth_answer,
                llm_provider=llm_provider,
                llm_kwargs=llm_kwargs
            )
            
            if not scores:
                logger.error("No valid scores found")
                raise ValueError("No valid scores found")
            
            # Update the scores in the strategies tracker
            for name, score in scores.items():
                strategies_tracker[name]['score'] = score
            
            # Select the best strategy based on the scores
            best_strategy_name = max(scores, key=scores.get)
            best_strategy_dict = strategies_tracker[best_strategy_name]

            # Create a new node for the best strategy
            new_node = ReasoningNode(
                strategy=best_strategy_dict['strategy'],
                prompt=best_strategy_dict['prompt'],
                response=best_strategy_dict['response'],
                cot=best_strategy_dict['cot'],
                metadata={"strategies_tracker": strategies_tracker}
            )
            beam.add_child(new_node)
            new_node.parent = beam
            beams[i] = new_node
            
            # Check if the new node is a terminal node with verifier
            verification_result, explanation = verifier(
                node=new_node,
                question=question,
                ground_truth_answer=ground_truth_answer,
                llm_provider=llm_provider,
                llm_kwargs=llm_kwargs or {}
            )
                
            # Append the verification explanation to the node's cot
            new_node.cot.append({"action": "verification", "content": explanation})
        
            # If verification is successful, mark the node as final
            # and add it to the terminal nodes
            if verification_result:
                new_node.is_success = True
                new_node.is_final = True
                continue
            
    result = SearchResult(
        final_node=None,
        all_terminal_nodes=beams,
        success=any(node.is_success for node in beams),
        final_answer=None,
        metadata={}
    )
    
    return result
