"""
The reasoning module provides components for building and exploring chains of thought (CoT).

Key Components:
- strategies: Defines Strategy classes that represent different reasoning approaches
- search: Implements search algorithms for exploring reasoning paths
- prompts: Contains prompt templates used by strategies
- cot_builder: Main interface for constructing chains of thought

Example usage:
```python
from cot_forge.llm import OpenAIProvider
from cot_forge.reasoning import CoTBuilder, Search, default_strategy_registry

llm = OpenAIProvider(api_key="...")
search = Search() 
builder = CoTBuilder(
    llm=llm,
    search=search,
    strategy_reg=default_strategy_registry
)

cot = builder.build(
    question="What is the capital of France?",
    ground_truth="Paris"
)
```
"""

from .cot_builder import CoTBuilder
from .search.naive_linear_search import naive_linear_search
from .search.search_algorithm import SearchAlgorithm, SearchResult
from .strategies import Strategy, StrategyRegistry, default_strategy_registry

__all__ = [
    "SearchAlgorithm",
    "SearchResult",
    "naive_linear_search",
    "Strategy",
    "StrategyRegistry",
    default_strategy_registry,
    "CoTBuilder",
    naive_linear_search
]