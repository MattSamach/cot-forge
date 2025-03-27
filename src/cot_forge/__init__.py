"""
CoT Forge: A library for constructing Chains of Thought (CoTs) using LLMs.

This package provides tools for reasoning with language models, including:
- Search algorithms (e.g., beam search)
- Verifiers for validating reasoning paths
- Scorers for evaluating reasoning quality
- Strategy registries for managing reasoning strategies

Modules:
    - reasoning: Core reasoning components
        - search: Search algorithms for reasoning
        - verifiers: Verifiers for validating reasoning paths
        - scorers: Scorers for evaluating reasoning quality
        - strategies: Strategy registries for managing reasoning strategies
    - llm: Language model interfaces
    - utils: Utility functions for reasoning and search
"""

#TODO: Try to move all imports to here but figure out circular imports first

# Core reasoning components
# LLM Provider exports
from .llm import GeminiProvider, LLMProvider
from .reasoning import (
                        CoTBuilder,
                        NaiveLinearSearch,
                        ReasoningNode,
                        SearchAlgorithm,
                        SearchResult,
                        SimpleBeamSearch,
)

# Scorer exports
from .reasoning.scorers import BaseScorer, ProbabilityFinalAnswerScorer

# Strategy-related exports
from .reasoning.strategies import (
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

# Verifier exports
from .reasoning.verifiers import BaseVerifier, LLMJudgeVerifier

# Utility functions
from .utils import (
                        execute_with_fallback,
                        extract_cot,
                        extract_final_answer_from_cot,
                        extract_final_answer_from_str,
                        generate_and_parse_cot,
                        generate_and_parse_json,
                        parse_json_response,
)

__all__ = [
    # Core components
    "CoTBuilder",
    "SearchAlgorithm",
    "SearchResult", 
    "NaiveLinearSearch",
    "SimpleBeamSearch",
    "ReasoningNode",
    
    # Strategies
    "Strategy",
    "StrategyRegistry",
    "default_strategy_registry",
    "AnalogicalReasoning",
    "Backtrack",
    "Correction", 
    "Counterfactual",
    "Decomposition",
    "ExploreNewPaths",
    "FirstPrinciples",
    "InitializeCoT",
    "PerspectiveShift",
    "Validation",
    
    # Utilities
    "extract_cot",
    "extract_final_answer_from_cot",
    "extract_final_answer_from_str",
    "parse_json_response",
    "execute_with_fallback",
    "generate_and_parse_cot",
    "generate_and_parse_json",
    
    # LLM Providers
    "LLMProvider",
    "GeminiProvider",
    
    # Verifiers
    "BaseVerifier", 
    "LLMJudgeVerifier",
    
    # Scorers
    "BaseScorer",
    "ProbabilityFinalAnswerScorer",
]
