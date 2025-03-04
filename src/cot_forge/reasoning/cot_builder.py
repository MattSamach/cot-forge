"""
Implementation of the CoTBuilder class, which is the main abstraction which
is used to handle the control flow to create CoT using sampling and search.
"""

from cot_forge.llm import LLMProvider

from .search import Search
from .strategies import StrategyRegistry


class CoTBuilder:
    """
    This class contains states and logic to use LLMs to construct 
    chains of thought (CoT) that, through reasoning, connect premises
    to ground truth answers.
    
    Args:
        search: Search algorithm used to construct chain of thought
        strategy_registry: Registry of strategies that can be sampled from
    """
    
    def __init__(self,
                 llm: LLMProvider,
                 search: Search,
                 strategy_reg: StrategyRegistry):
        
        self.cot = []
        self.llm = llm
        self.strategy_reg = strategy_reg
        self.search = search
        