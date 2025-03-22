"""
This module defines abstract and concrete strategy selectors for choosing reasoning strategies.

It provides mechanisms for selecting appropriate reasoning strategies during chain-of-thought
(CoT) generation. The module includes base interfaces for strategy selection and implements
different selection approaches ranging from simple random selection to sophisticated scoring-based
selection.

Classes:
    StrategySelector: Abstract base class defining the interface for strategy selectors.
    RandomStrategySelector: Selector that randomly chooses strategies from available options.
    ScoredStrategySelector: Selector that ranks strategies using a provided scoring function.

The strategy selection process typically considers:
    1. The current depth in the reasoning chain
    2. Whether this is an initial or continuation step
    3. Minimum depth requirements of available strategies
    4. Optional scoring mechanisms to evaluate strategy quality

Usage example:
    ```python
    from cot_forge.reasoning.strategies import StrategyRegistry, default_strategy_registry
    from cot_forge.reasoning.strategies.strategy_selector import RandomStrategySelector
    
    # Create a selector
    selector = RandomStrategySelector()
    
    # Select a strategy at depth 0 (initial)
    strategies, info = selector.select(
        registry=default_strategy_registry,
        depth=0,
        num_strategies=1
    )
    ```
"""

import random
from abc import ABC, abstractmethod
from collections import defaultdict
from logging import Logger
from typing import TYPE_CHECKING, Any

from cot_forge.llm import LLMProvider
from cot_forge.reasoning.scorers import BaseScorer
from cot_forge.utils.search_utils import generate_and_parse_cot

from .strategies import Strategy, StrategyRegistry

if TYPE_CHECKING:
    from cot_forge.reasoning.types import ReasoningNode

logger = Logger(__name__)

class StrategySelector(ABC):
    """
    Abstract base class defining the interface for strategy selection algorithms.
    
    Strategy selectors are responsible for choosing appropriate reasoning strategies
    based on the current state of the reasoning process, such as the depth in the
    reasoning chain and available strategy options.
    """
    
    @abstractmethod
    def select(self,
               registry: StrategyRegistry,
               depth: int,
               num_strategies: int = 1,
               **kwargs) -> tuple[list[Strategy], dict[str, Any]]:
        """
        Select strategies based on the provided registry and current reasoning state.
        
        This abstract method must be implemented by concrete selector classes to define
        the specific selection algorithm.
        
        Args:
            registry: The strategy registry containing available strategies.
            depth: The current depth in the reasoning chain.
            num_strategies: The number of strategies to select.
            **kwargs: Additional arguments for specific implementations.
            
        Returns:
            tuple: A tuple containing:
                - list[Strategy]: A list of selected strategy objects.
                - dict[str, Any]: A dictionary with optional additional information about the selection.
        """
        pass
    
    def get_strategy_options(
        self,
        registry: StrategyRegistry,
        depth: int,
        num_considered: int = None,
    ) -> list[Strategy]:
        """
        Get a list of possible strategies based on depth and registry constraints.
        
        This method filters the available strategies to ensure they meet requirements
        for the current reasoning state (e.g., initial vs continuation, minimum depth).
        
        Args:
            registry: The strategy registry containing available strategies.
            depth: The current depth in the reasoning chain.
            num_considered: The maximum number of strategies to consider. If None,
                all eligible strategies are considered.
            
        Returns:
            list[Strategy]: A list of strategy objects appropriate for the current depth.
            
        Raises:
            ValueError: If no appropriate strategies are found for the current state.
        """
        
        strategy_names = registry.list_strategies()
        
        # Filter out initial strategies if not the first step
        if depth == 0:
            # First step must be the initial strategy
            strategy = registry.get_strategy("initialize")
            return [strategy]
        else:
            # Exclude initial strategies and those not meeting minimum depth
            strategy_names = [
                name for name in strategy_names 
                if not registry.get_strategy(name).is_initial
                and depth >= registry.get_strategy(name).minimum_depth
            ]
            
        # If no limit is set, consider all strategies
        if num_considered is None:
            num_considered = len(strategy_names)
            
        # Wittle down to the number of strategies to consider
        strategy_names = random.sample(strategy_names, num_considered)
            
        if not strategy_names:
            step_type = "initial" if depth == 0 else "continuation"
            raise ValueError(f"No appropriate strategies found for {step_type} step")
        
        return [registry.get_strategy(name) for name in strategy_names]
    
    def __str__(self) -> str:
        """Return a string representation of the strategy selector."""
        return f"{self.__class__.__name__}()"

    def __repr__(self) -> str:
        """Return a detailed string representation of the strategy selector."""
        return f"{self.__class__.__name__}()"
    
class RandomStrategySelector(StrategySelector):
    """Randomly selects a strategy from the registry."""
    
    def select(
        self,
        registry: StrategyRegistry,
        depth: int, 
        num_strategies: int = 1, 
        **kwargs
    ) -> tuple[list[Strategy], dict[str, Any]]:
        """
        Select random strategies from the eligible options in the registry.
        
        Args:
            registry: The strategy registry containing available strategies.
            depth: The current depth in the reasoning chain.
            num_strategies: The number of strategies to select.
            **kwargs: Unused additional arguments.
            
        Returns:
            tuple: A tuple containing:
                - list[Strategy]: A list of randomly selected strategy objects.
                - dict: An empty dictionary as no additional selection info is provided.
                
        Raises:
            ValueError: If no eligible strategies are found or if requesting more
                strategies than are available.
        """
        
        strategy_options = self.get_strategy_options(registry, depth)
        selected_strategies = random.sample(strategy_options, num_strategies)
        logger.debug(f"Selected strategies: {[strategy.name for strategy in strategy_options]}")
        return selected_strategies, {}
    
class ScoredStrategySelector(StrategySelector):
    """
    A strategy selector that ranks strategies based on a scoring function.
    
    This selector generates CoTs using each eligible strategy, scores them using
    a provided scorer, and selects the highest-scoring strategies. This
    allows for more sophisticated strategy selection based on the quality of the
    reasoning each strategy produces.
    """
    
    def select(
        self,
        search_llm: LLMProvider,
        registry: StrategyRegistry,
        depth: int,
        question: str,
        ground_truth_answer: str,
        scorer: BaseScorer,
        num_strategies: int = 1,
        num_considered: int = None,
        node: 'ReasoningNode' = None,
        llm_kwargs: dict[str, Any] = None,
        **kwargs
    ) -> tuple[list[Strategy], dict[str, Any]]:
        """
        Select strategies by scoring their performance on the given question.
        
        This method:
        1. Generates a reasoning chain using each eligible strategy
        2. Scores the resulting chains using the provided scorer
        3. Selects the top-performing strategies based on these scores
        
        Args:
            search_llm: The LLM provider for generating reasoning chains.
            registry: The strategy registry containing available strategies.
            depth: The current depth in the reasoning chain.
            question: The question being addressed in the reasoning process.
            ground_truth_answer: The expected answer for scoring accuracy.
            scorer: The scorer (BaseScorer) to evaluate strategy performance.
            num_strategies: The number of strategies to select.
            num_considered: The maximum number of strategies to evaluate. 
                If None, all eligible strategies are considered.
            node: The current reasoning node to which new reasoning will be appended.
            llm_kwargs: Additional arguments for the LLM provider.
            **kwargs: Unused additional arguments.
            
        Returns:
            tuple: A tuple containing:
                - list[Strategy]: A list of the highest-scoring strategy objects.
                - dict: A dictionary containing:
                    - "strategies_dict": Detailed information about each strategy evaluation.
                    - "scores": The scores assigned to each strategy.
                    
        Raises:
            ValueError: If no eligible strategies are found or if no strategies 
                could be successfully scored.
        """
        strategy_options = self.get_strategy_options(registry, depth, num_considered=num_considered)
        llm_kwargs = llm_kwargs or {}
        
        # Generate and parse COT for each strategy
        strategies_dict = defaultdict(dict)
        for strategy in strategy_options:
            prompt = strategy.build_prompt(
                question=question,
                previous_cot=str(node.get_full_cot() if node else None)
            )
            try:
                response, cot = generate_and_parse_cot(
                    search_llm,
                    prompt=prompt,
                    llm_kwargs=llm_kwargs,
                    on_error="raise"
                )
            except Exception as e:
                logger.error(f"Error generating COT for strategy {strategy.name}: {e}")
                continue
            
            # Store strategy, response, and cot in the dictionary
            strategies_dict[strategy.name]["prompt"] = prompt
            strategies_dict[strategy.name]["strategy"] = strategy
            strategies_dict[strategy.name]["response"] = response
            strategies_dict[strategy.name]["cot"] = cot
            
        if not strategies_dict:
            raise ValueError("No strategies available for scoring.")
        
        # Score each strategy
        # Create cot_list in the format expected by the scorer
        cot_list = [
            {"strategy_name": strat_name, "cot": strat_data['cot']}
            for strat_name, strat_data in strategies_dict.items()
        ]
        
        # Use scorer to score the strategies
        try:
            scores = scorer(
                cot_list=cot_list,
                question=question,
                ground_truth_answer=ground_truth_answer,
            )
        except Exception as e:
            logger.error(f"Error in scoring: {e}")
            return None
        
        # Update strategies_dict with scores
        for strat_name, score in scores.items():
            strategies_dict[strat_name]["score"] = score
            
        # Select the best strategies based on the scores
        scores = {strat: strategies_dict[strat]['score'] for strat in strategies_dict}
        sorted_strategies = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        
        # Select the top strategies
        # If num_strategies is greater than the number of available strategies, loop back to the start
        i = 0
        selected_strategy_names = []
        while len(selected_strategy_names) < num_strategies:
            strat_name = sorted_strategies[i % len(sorted_strategies)][0]
            selected_strategy_names.append(strat_name)
            i += 1
            
        # Convert strategy names to Strategy objects
        selected_strategies = [registry.get_strategy(strat_name) for strat_name in selected_strategy_names]
        logger.debug(f"Selected strategies: {[strategy_name for strategy_name in selected_strategy_names]}")
        
        return selected_strategies, {
            "strategies_dict": strategies_dict,
            "scores": scores,
        }