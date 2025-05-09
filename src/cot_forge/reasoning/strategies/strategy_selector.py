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
             **kwargs) -> dict[str, Any]:
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
        - dict[str, Any]: A dictionary cotaining:
            - "selected_strategies": The list of selected strategies.
            - Any additional information relevant to the selection process.
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
  ) -> dict[str, Any]:
    """
    Select random strategies from the eligible options in the registry.

    Args:
        registry: The strategy registry containing available strategies.
        depth: The current depth in the reasoning chain.
        num_strategies: The number of strategies to select.
        **kwargs: Unused additional arguments.

    Returns:
        - dict: A dictionary containing:
            - "selected_strategies": The list of selected strategies.

    Raises:
        ValueError: If no eligible strategies are found or if requesting more
            strategies than are available.
    """

    strategy_options = self.get_strategy_options(registry, depth)
    selected_strategies = random.sample(strategy_options, num_strategies)
    logger.debug(
        f"Selected strategies: {[strategy.name for strategy in strategy_options]}")
    return {"selected_strategies": selected_strategies}


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
      nodes: list['ReasoningNode'],
      num_strategies: int = 1,
      num_considered: int = None,
      llm_kwargs: dict[str, Any] = None,
      **kwargs
  ) -> list[dict[str, Any]]:
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
        nodes: The list of reasoning nodes to consider to append to.
        num_strategies: The number of strategies to select.
        num_considered: The maximum number of strategies to evaluate. 
            If None, all eligible strategies are considered.
        llm_kwargs: Additional arguments for the LLM provider.
        **kwargs: Unused additional arguments.

    Returns:
        - list[dict[str, Any]]: A list of dictionaries.
            Each index correspondes to the node in the parameter `nodes` list.
            Each dictionary contains:
            - "strategy": The strategy object.
            - "prompt": The generated prompt for the strategy.
            - "response": The generated response from the LLM.
            - "cot": The generated chain of thought.
            - "score": The score assigned by the scorer.
            - "selection_count": The number of times this strategy was selected.
            - "option_id": The unique identifier for the strategy option.

    Raises:
        ValueError: If no eligible strategies are found or if no strategies 
            could be successfully scored.
    """

    llm_kwargs = llm_kwargs or {}

    # List storing strategy options dicts for each node
    strategy_options_by_node = []
    # Dictionary to remmap options to nodes
    option_node_map = {}
    i = 0

    # Loop through nodes
    for node_idx, node in enumerate(nodes):
      # Do not need to select strategies if the node is final
      if node.is_final:
        strategy_options_by_node.append({})
        continue
      strategy_options = self.get_strategy_options(
          registry, depth, num_considered=num_considered)
      if not strategy_options:
        raise ValueError("No strategies available for for scoring.")

      # Generate and parse COT for each strategy
      strategies_dict = defaultdict(dict)
      for strategy in strategy_options:
        prompt = strategy.build_prompt(
            question=question,
            previous_cot=str(node.get_full_cot())
        )
        try:
          response, cot = generate_and_parse_cot(
              search_llm,
              prompt=prompt,
              llm_kwargs=llm_kwargs
          )
        except Exception as e:
          logger.error(
              f"Error generating COT for strategy {strategy.name}: {e}")
          continue

        # Store strategy, response, and cot in the dictionary
        option_id = f"option_{i}"
        strategies_dict[option_id]["strategy"] = strategy
        strategies_dict[option_id]["prompt"] = prompt
        strategies_dict[option_id]["response"] = response
        strategies_dict[option_id]["cot"] = cot
        strategies_dict[option_id]["selection_count"] = 0
        i += 1

        # Map option_id to node index
        option_node_map[option_id] = node_idx

      # Append to strat options at corresponding node index
      strategy_options_by_node.append(strategies_dict)

    # Score each strategy
    # Create cot_list in the format expected by the scorer
    cot_list = [
        {'option_id': option_id, 'cot': option['cot']}
        for strat_dict in strategy_options_by_node
        for option_id, option in strat_dict.items()
    ]

    # Use scorer to score the strategies
    try:
      scores = scorer(
          cot_list=cot_list,
          question=question,
          ground_truth_answer=ground_truth_answer,
          id_field="option_id",
      )
    except Exception as e:
      logger.error(f"Error in scoring: {e}")
      return None

    # Use option_node_map to map scores back to nodes
    for option_id, score in scores.items():
      node_idx = option_node_map.get(option_id)
      if node_idx is not None:
        # Update the score in the corresponding strategy dict
        strategy_options_by_node[node_idx][option_id]["score"] = score

    sorted_options = sorted(
        scores.items(), key=lambda x: x[1], reverse=True
    )

    # Select the top options based on the scores
    # If num_strategies is greater than the number of available options, loop back to the start
    i = 0
    selected_options = []
    while len(selected_options) < num_strategies:
      selected_options.append(sorted_options[i % len(sorted_options)][0])
      i += 1

    # Increment the selection count for each selected option
    # Options may be selected multiple times if num_strategies > len(sorted_options)
    # That would mean the same option selected to branch from the same node
    for option_id in selected_options:
      node_idx = option_node_map.get(option_id)
      if node_idx is not None:
        strategy_options_by_node[node_idx][option_id]["selection_count"] += 1

    return strategy_options_by_node
