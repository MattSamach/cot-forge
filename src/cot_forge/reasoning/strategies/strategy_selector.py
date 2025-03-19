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
    """Base class for strategy selection algorithms."""
    
    @abstractmethod
    def select(self,
               registry: StrategyRegistry,
               depth: int,
               num_strategies: int = 1,
               **kwargs) -> tuple[list[Strategy], dict[str, Any]]:
        """
        Select  based on the provided registry and depth.
        
        Args:
            registry: The strategy registry.
            depth: The current depth in the reasoning chain.
            num_strategies: The number of strategies to select.
            **kwargs: Additional arguments for specific implementations.
            
        Returns:
            Tuple containing:
                - A list of selected strategies.
                - A dictionary with optional additional information about the selection.
        """
        pass
    
    def get_strategy_options(
        self,
        registry: StrategyRegistry,
        depth: int,
        num_considered: int = None,
    ) -> list[Strategy]:
        """Get list of possible strategy names based on depth and registry.
        
        Args:
            registry: The strategy registry.
            depth: The current depth in the reasoning chain.
            num_considered: The number of strategies to consider for scoring. (Each require a separate cot generation call to LLM)
            
        Returns:
            A list of strategies appropriate for the current depth.
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
        Select a random strategy from the registry.
        
        Args:
            registry: The strategy registry.
            depth: The current depth in the reasoning chain.
            
        Returns:
            A randomly selected strategy.
        """
        """Select a random strategy from the registry.
        
        Args:
            registry: The strategy registry.
            depth: The current depth in the reasoning chain.
            num_strategies: The number of strategies to select.
            **kwargs: Additional arguments for specific implementations.
            
        Returns:
            A randomly selected strategy.
        """
        
        strategy_options = self.get_strategy_options(registry, depth)
        selected_strategies = random.sample(strategy_options, num_strategies)
        logger.debug(f"Selected strategies: {[strategy.name for strategy in strategy_options]}")
        return selected_strategies, {}
    
class ScoredStrategySelector(StrategySelector):
    """Selects the best strategy based on scorer."""
    
    def select(
        self,
        reasoning_llm: LLMProvider,
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
        Select the best strategy based on the provided scorer.
        
        Args:
            reasoning_llm: The reasoning LLM provider.
            registry: The strategy registry.
            depth: The current depth in the reasoning chain.
            question: The question being asked.
            ground_truth_answer: The expected answer.
            scorer: The scoring function to use.
            num_strategies: The number of strategies to select.
            num_considered: The number of strategies to consider for scoring. (Each require a separate cot generation call to LLM)
            node: The current reasoning node on which we want to append the selected strategy cot.
            llm_kwargs: Additional arguments for the reasoning LLM provider.
            **kwargs: Additional arguments for specific implementations.
            
        Returns:
            Tuple containing:
                - A list of selected strategies.
                - A dictionary with optional additional information about the selection.
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
                    reasoning_llm,
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