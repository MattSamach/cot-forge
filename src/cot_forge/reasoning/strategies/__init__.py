"""
The `strategies` module provides a framework for implementing and managing reasoning strategies,
including prompt templates, concrete strategy implementations, a strategy registry, and strategy selectors.

It defines core classes like `Strategy` and `StrategyPromptTemplate`, along with various concrete
reasoning strategies such as `Backtrack`, `Correction`, and `ExploreNewPaths`.  It also includes
utilities for selecting strategies, such as `RandomStrategySelector` and `ScoredStrategySelector`.
"""

from .prompts import (
                      StrategyPromptTemplate,
                      backtrack_strategy_prompt,
                      correction_strategy_prompt,
                      explore_new_paths_strategy_prompt,
                      initialize_cot_prompt,
                      validation_strategy_prompt,
)
from .strategies import (
                      AnalogicalReasoning,
                      Backtrack,
                      Correction,
                      Counterfactual,
                      Decomposition,
                      ExploreNewPaths,
                      FirstPrinciples,
                      InitializeCoT,
                      PerspectiveShift,
                      Strategy,
                      StrategyRegistry,
                      Validation,
                      default_strategy_registry,
)
from .strategy_selector import RandomStrategySelector, ScoredStrategySelector, StrategySelector

__all__ = [
    "Strategy",
    "StrategyPromptTemplate",
    "StrategyRegistry",
    "InitializeCoT",
    "Backtrack",
    "ExploreNewPaths",
    "Correction",
    "Validation",
    "Counterfactual",
    "Decomposition",
    "AnalogicalReasoning",
    "FirstPrinciples",
    "PerspectiveShift",
    "RandomStrategySelector",
    "StrategySelector",
    "ScoredStrategySelector",
    default_strategy_registry,
    initialize_cot_prompt,
    backtrack_strategy_prompt,
    explore_new_paths_strategy_prompt,
    correction_strategy_prompt,
    validation_strategy_prompt
]
