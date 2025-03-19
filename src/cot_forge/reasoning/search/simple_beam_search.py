"""
Beam search implementation for reasoning chains.
This search algorithm expands the reasoning chain by maintaining multiple paths
and selecting the most promising ones based on a scoring mechanism.
Branching factor options are chosen at random.

Logical flow:
1. Initialize the chain with an initial cot.
2. Initialize beams with distinct strategies (only reuse if beam_width > number_available_strategies).
3. Generate the next step in the reasoning chain using a strategy.
4. Use a scoring mechanism to select the most promising paths.
5. Check if the generated chain is valid using a verifier. If valid, mark it as final.
6. Expand each beam by choosing the best strategy for the current node.
7. Continue until a termination condition is met or max depth is reached.
8. Return all beams.
"""
# TODO: Experiment with multithreading in beam generation

import logging
from typing import Any

from cot_forge.llm import LLMProvider
from cot_forge.reasoning.scorers import BaseScorer
from cot_forge.reasoning.strategies import (
    InitializeCoT,
    ScoredStrategySelector,
    StrategyRegistry,
    default_strategy_registry,
)
from cot_forge.reasoning.types import ReasoningNode, SearchResult
from cot_forge.reasoning.verifiers import BaseVerifier
from cot_forge.utils.search_utils import generate_and_parse_cot

from .search_algorithm import BaseSearch

logger = logging.getLogger(__name__)

class SimpleBeamSearch(BaseSearch):
    """
    Simple beam search to produce multiple parallel reasoning chains.
    
    This class implements a simple beam search algorithm to generate multiple reasoning chains
    in parallel. It uses a scoring mechanism to evaluate the generated strategies and selects the
    most promising paths for further exploration.
    
    Attributes:
        strategy_registry (StrategyRegistry): The strategy registry to use for selecting strategies.
        beam_width (int): Number of beams to be explored.
        branching_factor (int): Number of strategies to consider at each node when extending each beam.
        max_depth (int): Maximum depth for the search.
        name (str): Name of the search algorithm.
        description (str): Description of the search algorithm.
        strategy_selector (ScoredStrategySelector): Strategy selector for scoring strategies.
    """
    
    def __init__(self,
                 beam_width: int = 2,
                 branching_factor: int = 3,
                 max_depth: int = 3,
                 strategy_registry: StrategyRegistry = default_strategy_registry,
                 ):
        self.beam_width = beam_width
        self.branching_factor = branching_factor
        self.max_depth = max_depth
        self.strategy_registry = strategy_registry
        self.name = "simple beam search"
        self.description = "Simple beam search to produce multiple parallel reasoning chains."
        self.strategy_selector = ScoredStrategySelector()

    def initialize_cot(
        self,
        question: str,
        ground_truth_answer: str,
        verifier: BaseVerifier,
        reasoning_llm: LLMProvider,
        llm_kwargs: dict[str, Any] = None
    ) -> ReasoningNode:
        """
        Returns a reasoning node with the initial CoT question and response.

        Args:
            question : The question
            ground_truth_answer: The true answer to the question
            verifier: The verifier to check correctness of final answers
            reasoning_llm: The LLM provider to generate initial CoT
            llm_kwargs: Additional kwargs for LLM provider

        Returns:
            ReasoningNode: Initial CoT reasoning node
        """
        strategy = InitializeCoT
        prompt = strategy.build_prompt(question)
        
        response, cot = generate_and_parse_cot(
            reasoning_llm=reasoning_llm,
            prompt=prompt,
            llm_kwargs=llm_kwargs,
            logger=logger,
            on_error="retry",
            max_retries=3
        )
        
        initial_node = self.create_node(
            strategy=strategy,
            prompt=prompt,
            response=response,
            cot=cot,
            metadata={"is_initial": True}
        )

        self.verify_node(
            node=initial_node,
            question=question,
            ground_truth_answer=ground_truth_answer,
            verifier=verifier,
            on_error='continue',
            logger=logger
        )
        
        # DEBUG
        # current_node.is_final = False
        # current_node.success = False

        return initial_node
    
    def initialize_beams(self,
                         initial_node: ReasoningNode,
                         strategy_registry: StrategyRegistry,
                         scorer: BaseScorer,
                         depth: int,
                         reasoning_llm: LLMProvider,
                         question: str,
                         ground_truth_answer: str,
                         verifier: BaseVerifier,
                         llm_kwargs: dict[str, Any] = None,
                         )-> list[ReasoningNode]:
        """
        Initialize the beams for the beam search by creating [beam_width] nodes. Each initialized beam 
        should try to employ different strategies to explore the search space.
        Args:
            initial_node: First node created. beam_width beams will be created from this node.
            strategy_registry: Registry of available strategies.
            scorer: Scorer to evaluate the strategies.
            depth: Current depth in the search.
            reasoning_llm: LLM provider to generate responses.
            question: The question to answer.
            ground_truth_answer: The true answer to the question.
            verifier: Verifier to check correctness of final answers.
            llm_kwargs: Additional kwargs for LLM provider.
        """
        
        llm_kwargs = llm_kwargs or {}

        try:
            selected_strategies, search_data = self.strategy_selector.select(
                reasoning_llm=reasoning_llm,
                registry=strategy_registry,
                depth=depth,
                num_strategies=self.beam_width,
                node=initial_node,
                question=question,
                ground_truth_answer=ground_truth_answer,
                scorer=scorer,
                llm_kwargs=llm_kwargs,
                logger=logger
            )
        except Exception as e:
            logger.error("Error in selecting strategies")
            raise ValueError("Failed to select strategies") from e
        
        strategies_dict, scores = search_data['strategies_dict'], search_data['scores']
        # Create beams with initial node as parent
        beams = []
        for strategy in selected_strategies:
            strat_data = strategies_dict[strategy.name]
            # Create a new node for each selected strategy
            new_beam = self.create_node(
                strategy=strat_data['strategy'],
                prompt=strat_data['prompt'],
                response=strat_data['response'],
                cot=strat_data['cot'],
                parent=initial_node,
                metadata={"is_initial": False, "scores": scores}
            )
            
            beams.append(new_beam)
            
        # Check if the new node is a terminal node with verifier
        for beam in beams:
            self.verify_node(
                node=beam,
                question=question,
                ground_truth_answer=ground_truth_answer,
                verifier=verifier,
                on_error="retry",
                logger=logger
            )

        return beams

    def _search(
        self,
        question: str,
        ground_truth_answer: str,
        reasoning_llm: LLMProvider,
        scorer: BaseScorer,
        verifier: BaseVerifier,
        strategy_registry: StrategyRegistry = default_strategy_registry,
        llm_kwargs: dict[str, Any] = None,
        max_depth: int = 3,
        beam_width: int = 3,
    ) -> SearchResult:
        """
        Perform a beam search to generate possible chains of thought.
        
        Args:
            question: The question to answer.
            ground_truth_answer: The true answer to the question.
            reasoning_llm: The LLM provider to use.
            scorer: The scorer used to evaluate different beam options.
            verifier: The verifier to use for checking correctness of final answers.
            strategy_registry: Registry of available strategies.
            llm_kwargs: Additional kwargs for reasoning LLM calls.
            strategy_registry: Registry of available strategies.
            max_depth: Maximum depth of the search tree (default: 3).
            beam_width: Number of beams to maintain at each step (default: 3).
            **kwargs: Additional kwargs for search algorithm.
            
        Returns:
            A SearchResult containing terminal nodes of beams.
        """
        # Create initial node
        try:
            initial_node = self.initialize_cot(
                question=question,
                ground_truth_answer=ground_truth_answer,
                verifier=verifier,
                reasoning_llm=reasoning_llm,
                llm_kwargs=llm_kwargs
            )
        except Exception as e:
            logger.error(f"Error in initializing CoT: {e}")
            return SearchResult(all_terminal_nodes=None, 
                                question=question,
                                ground_truth_answer=ground_truth_answer,
                                success=False,
                                metadata={"error": "Failed to initialize CoT",
                                          "max_depth": max_depth,
                                          "question": question,
                                          "ground_truth_answer": ground_truth_answer}
                                )
            
        # Check if initial node is already successful
        initial_node.success = False
        if initial_node.success:
            return SearchResult(
                final_node=initial_node,
                all_terminal_nodes=[initial_node],
                success=True,
                final_answer=initial_node.response,
                metadata={"max_depth": max_depth,
                          "question": question,
                          "ground_truth_answer": ground_truth_answer}
            )
            
        # Initialize the beams
        try:
            beams = self.initialize_beams(
                initial_node=initial_node,
                strategy_registry=strategy_registry,
                scorer=scorer,
                depth=1,
                reasoning_llm=reasoning_llm,
                question=question,
                ground_truth_answer=ground_truth_answer,
                verifier=verifier,
                llm_kwargs=llm_kwargs,
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
                
                try:
                    selected_strategies, search_data = self.strategy_selector.select(
                        reasoning_llm=reasoning_llm,
                        registry=strategy_registry,
                        depth=depth,
                        question=question,
                        ground_truth_answer=ground_truth_answer,
                        scorer=scorer,
                        node=beam,
                        llm_kwargs=llm_kwargs,
                        logger=logger,
                    )
                except Exception as e:
                    logger.error("Error in selecting strategies")
                    raise ValueError("Failed to select strategies") from e
                
                strategies_dict, scores = search_data['strategies_dict'], search_data['scores']
                
                # If strategies_tracker is None, consider it a failure
                # and skip this beam at this depth
                if strategies_dict is None:
                    continue
                
                best_strategy = selected_strategies[0]
                best_strategy_dict = strategies_dict[best_strategy.name]
                
                # Create a new node for the best strategy
                new_node = self.create_node(
                    strategy=best_strategy_dict['strategy'],
                    prompt=best_strategy_dict['prompt'],
                    response=best_strategy_dict['response'],
                    cot=best_strategy_dict['cot'],
                    parent=beam,
                    metadata={"is_initial": False, "scores": scores}
                )
               
                beams[i] = new_node
                
                # Check if the new node is a terminal node with verifier
                # If the verification fails, continue with the beam search
                verification_result, error = self.verify_node(
                    node=new_node,
                    question=question,
                    ground_truth_answer=ground_truth_answer,
                    verifier=verifier,
                    on_error='continue',
                    logger=logger
                )
                # If verification fails, skip this node at this depth
                if error is not None:
                    continue

                # If verification is successful, log the result
                if verification_result:
                    logger.info(f"Beam {i} reached a final node at depth {depth}")
                
        result = SearchResult(
            final_node=None,
            all_terminal_nodes=beams,
            success=any(node.success for node in beams),
            final_answer=None,
            metadata={"max_depth": max_depth,
                      "question": question,
                      "ground_truth_answer": ground_truth_answer}
        )
        
        return result
