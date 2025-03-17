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
from cot_forge.reasoning.verifiers import BaseVerifier
from cot_forge.utils.parsing import extract_cot
from cot_forge.utils.search_utils import try_operation

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

    def verify_node(
        self,
        node: ReasoningNode, 
        question: str, 
        ground_truth_answer: str, 
        verifier: BaseVerifier,
        init_phase: bool = False
    ) -> tuple[bool, str | None]:
        """
        Verify a node and update its status.
        
        Args:
            node: The node to verify
            question: Original question
            ground_truth_answer: The true answer
            verifier: Verification function to use
            reasoning_llm: LLM provider for verification
            llm_kwargs: Additional kwargs for LLM provider
            init_phase: Whether this is during initial beam creation (affects error handling)
            
        Returns:
            tuple[bool, str | None]: (verification_success, error_message if any)
        """        
        def fallback_handler(exception, ctx):
            """Context-aware fallback for verification errors"""
            if ctx.get("init_phase", False):
                # During initialization, continue with default values
                return False, ""
            # During main search, return error to allow caller to skip
            return False, None
        
        # Execute verification with context-aware error handling
        result, error_msg = try_operation(
            "verification",
            lambda: verifier(
                node=node,
                question=question,
                ground_truth_answer=ground_truth_answer,
            ),
            fallback_function=fallback_handler,
            context={"init_phase": init_phase},
            logger=logger,
        )
        
        if error_msg and not init_phase:
            # During main search, immediately return on error
            return False, error_msg
        
        verification_result, explanation = result if result else (False, "")
        
        # Update node metadata with verification details
        node.metadata["verification"] = explanation
        
        # If verification is successful, mark the node as final and successful
        if verification_result:
            node.success = True
            node.is_final = True
        
        return verification_result, error_msg

    def evaluate_strategies(
        self,
        node: ReasoningNode,
        strategy_registry: StrategyRegistry,
        depth: int,
        num_strategies: int,
        reasoning_llm: LLMProvider,
        llm_kwargs: dict[str, Any],
        question: str,
        ground_truth_answer: str,
        scorer: BaseScorer
    ) -> dict[str, dict] | None:  # Modified to return None on failure
        """
        Evaluate multiple strategies for a given node and rank them.
        
        Args:
            node: Current node to expand from
            strategy_registry: Registry of available strategies
            depth: Current depth in the search
            num_strategies: Number of strategies to consider
            reasoning_llm: The LLM provider
            llm_kwargs: Additional kwargs for reasoning llm call
            question: The original question
            ground_truth_answer: The true answer
            scorer: Scoring function to evaluate strategies
            
        Returns:
            Dictionary of strategy evaluations with scores or None if scoring fails
        """
        strategy_options = self.get_strategy_options(
            registry=strategy_registry,
            num_strategies=num_strategies,
            depth=depth
        )
        
        # Generate CoT for each strategy option
        strategies_tracker = defaultdict(dict)
        for strat_name in strategy_options:
            strategy = strategy_registry.get_strategy(strat_name)
            # TODO: Move error handling to try_operation
            try:
                response, cot, prompt = self.generate_cot(
                    node=node,
                    strategy=strategy,
                    reasoning_llm=reasoning_llm,
                    llm_kwargs=llm_kwargs
                )
            except Exception as e:
                logger.error(f"Error in generating response: {e}")
                continue

            # Store the strategy, response, and cot in the strategies tracker
            strategies_tracker[strat_name]['response'] = response
            strategies_tracker[strat_name]['cot'] = cot
            strategies_tracker[strat_name]['prompt'] = prompt
            strategies_tracker[strat_name]['strategy'] = strategy_registry.get_strategy(strat_name)
            
        if not strategies_tracker:
            logger.error("No valid strategies found")
            return None  # Return None instead of raising error

        # Create cot_list in the format expected by the scorer
        cot_list = [
            {"strategy_name": strat_name, "cot": strat_data['cot']}
            for strat_name, strat_data in strategies_tracker.items()
        ]

        # Score each option using the provided scorer
        try:
            scores = scorer(
                cot_list=cot_list,
                question=question,
                ground_truth_answer=ground_truth_answer,
            )
        except Exception as e:
            logger.error(f"Error in scoring: {e}")
            return None  # Return None instead of raising error

        # Update scores
        for name, score in scores.items():
            strategies_tracker[name]['score'] = score
                
        return strategies_tracker

    def get_strategy_options(
        self,
        registry: StrategyRegistry,
        depth: int,
        num_strategies: int = None,
    ) -> list[Strategy]:
        """
        Selects a set of strategies options for a beam to consider.
        
        Args:
            registry: The strategy registry.
            depth: Current depth in the reasoning chain. Needed to filter strategies.
            num_strategies: Number of strategies to sample. If None, all strategies are considered.
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
        
        if num_strategies > len(strategy_names):
            raise ValueError("branching_factor cannot exceed number of valid strategies in registry")
        
        # Consider all strategies if no branching factor
        if num_strategies is None:
            num_strategies = len(strategy_names)
            
        return random.sample(strategy_names, num_strategies)

    def initialize_cot(
        self,
        question: str,
        ground_truth_answer: str,
        verifier: BaseVerifier,
        reasoning_llm: LLMProvider,
        llm_kwargs: dict[str, Any] = None
    ) -> ReasoningNode :
        """
        Returns a reasoning node with the initial CoT question and response.

        Args:
            question : The question
            reasoning_llm: The LLM provider to generate initial CoT

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
        response, error = try_operation(
            "LLM generation",
            reasoning_llm.generate,
            kwargs={"prompt": prompt, 'llm_kwargs': (llm_kwargs or {})},
            error_action="raise",
            logger=logger
        )
            
        current_node.response = response
        
        # Extract CoT from response
        cot, error = try_operation(
            "CoT extraction",
            extract_cot,
            args=(response,),
            error_action="raise",
            logger = logger
        )
        
        current_node.cot = cot
        
        self.verify_node(
            node=current_node,
            question=question,
            ground_truth_answer=ground_truth_answer,
            verifier=verifier,
            init_phase=True
        )
        
        # DEBUG
        # current_node.is_final = False
        # current_node.success = False

        return current_node


    def generate_cot(self,
                     node: ReasoningNode,
                     strategy: Strategy,
                     reasoning_llm: LLMProvider,
                     llm_kwargs: dict[str, Any] = None
                     ) -> tuple[str | None, dict[str, Any] | None, str]:
        """
        Generate a chain of thought using the provided strategy.
        Args:
            node: Current deepest node in the beam.
            strategy: Strategy to use for generating the next step.
            reasoning_llm: LLM provider to generate the response.
            llm_kwargs: Additional kwargs for reasoning llm call.
        Returns:
            (str, dict): Response and  dictionary containing the generated chain of thought.
        """
        prompt = strategy.build_prompt(
            question=node.prompt,
            previous_cot=str(node.get_full_cot())
        )
        
        # Generate response.
        response, error = try_operation(
            "LLM generation",
            reasoning_llm.generate,
            kwargs={"prompt": prompt, 'llm_kwargs': (llm_kwargs or {})},
            error_action="raise",
            logger = logger
        )
        
        cot, error = try_operation(
            "CoT extraction",
            extract_cot,
            args=(response,),
            error_action="raise",
            logger = logger
        )
        
        return response, cot, prompt
    
    def initialize_beams(self,
                         initial_node: ReasoningNode,
                         strategy_registry: StrategyRegistry,
                         scorer: BaseScorer,
                         depth: int,
                         reasoning_llm: LLMProvider,
                         question: str,
                         ground_truth_answer: str,
                         verifier: BaseVerifier,
                         llm_kwargs: dict[str, Any] = {},
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
        
        # Get strategy scores for the initial node
        try:
            strategies_tracker = self.evaluate_strategies(
                node=initial_node,
                strategy_registry=strategy_registry,
                depth=depth,
                num_strategies=self.beam_width,
                question=question,
                ground_truth_answer=ground_truth_answer,
                scorer=scorer,
                reasoning_llm=reasoning_llm,
                llm_kwargs=llm_kwargs
            )
        except Exception as e:
            logger.error(f"Error in evaluating strategies: {e}")
            raise ValueError("Failed to score strategies")
        
        # Select the best strategies based on the scores
        selected_strategies = []
        scores = {strat: strategies_tracker[strat]['score'] for strat in strategies_tracker}
        sorted_strategies = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        i = 0
        while len(selected_strategies) < self.beam_width:
            strategy_name, score = sorted_strategies[i]
            selected_strategies.append(strategy_name)
            i += 1
            if i >= len(sorted_strategies):
                i = 0
                
        # Create beams with initial node as parent
        beams = []
        for strategy_name in selected_strategies:
            strat_data = strategies_tracker[strategy_name]
            new_beam = ReasoningNode(
                strategy=strat_data['strategy'],
                prompt=strat_data['prompt'],
                response=strat_data['response'],
                cot=strat_data['cot'],
                parent=initial_node,
                metadata={"scores": scores}
            )
            initial_node.add_child(new_beam)
            beams.append(new_beam)
            
        # Check if the new node is a terminal node with verifier
        for beam in beams:
            self.verify_node(
                node=beam,
                question=question,
                ground_truth_answer=ground_truth_answer,
                verifier=verifier,
                init_phase=True
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
                                metadata={"error": "Failed to initialize CoT"})
            
        # Check if initial node is already successful
        if initial_node.success:
            return SearchResult(
                final_node=initial_node,
                all_terminal_nodes=[initial_node],
                success=True,
                final_answer=initial_node.response,
                metadata={"max_depth": max_depth, "question": question, "ground_truth_answer": ground_truth_answer}
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
                
                strategies_tracker = self.evaluate_strategies(
                    node=beam,
                    strategy_registry=strategy_registry,
                    depth=depth,
                    num_strategies=self.branching_factor,
                    reasoning_llm=reasoning_llm,
                    llm_kwargs=llm_kwargs,
                    question=question,
                    ground_truth_answer=ground_truth_answer,
                    scorer=scorer
                )
                
                # If strategies_tracker is None, consider it a failure
                # and skip this beam at this depth
                if strategies_tracker is None:
                    continue

                scores = {strat: strategies_tracker[strat]['score'] for strat in strategies_tracker}
                best_strategy_name = max(scores, key=scores.get)
                best_strategy_dict = strategies_tracker[best_strategy_name]

                # Create a new node for the best strategy
                new_node = ReasoningNode(
                    strategy=best_strategy_dict['strategy'],
                    prompt=best_strategy_dict['prompt'],
                    response=best_strategy_dict['response'],
                    cot=best_strategy_dict['cot'],
                    metadata={"scores": scores},
                )
                beam.add_child(new_node)
                new_node.parent = beam
                beams[i] = new_node
                
                # Check if the new node is a terminal node with verifier
                # If the verification fails, continue with the beam search
                verification_result, error = self.verify_node(
                    node=new_node,
                    question=question,
                    ground_truth_answer=ground_truth_answer,
                    verifier=verifier,
                    init_phase=False  # This is the main search phase
                )
                # If verification fails, skip this node at this depth
                if error:
                    continue
                
                # Update the node's metadata with verification details
                new_node.metadata["verification"] = verification_result
                
                # If verification is successful, log the result
                if verification_result:
                    logger.info(f"Beam {i} reached a final node at depth {depth}")
                
        result = SearchResult(
            final_node=None,
            all_terminal_nodes=beams,
            success=any(node.success for node in beams),
            final_answer=None,
            metadata={"max_depth": max_depth, "question": question, "ground_truth_answer": ground_truth_answer}
        )
        
        return result
