"""
This file defines strategies that constitute individual links within
the chain of thought (CoT). It provides a standard interface for these
strategies, and also defines a "strategies registry" that allows users
to define their own strategies and register them.
"""

from dataclasses import dataclass
from typing import Any, ClassVar, Optional

from . import prompts
from .prompts import StrategyPromptTemplate


@dataclass(frozen=True)
class Strategy:
    """
    Base strategy class with common attributes.
    
    Attributes:
        name (str): Name of the strategy.
        description (str): Description of the strategy.
        is_initial (bool): Indicates if this is an initial strategy.
    """
    name: ClassVar[str]
    description: ClassVar[str]
    is_initial: ClassVar[bool]
    
    @classmethod
    def get_metadata(cls) -> dict[str, Any]:
        return {"name": cls.name, 
                "description": cls.description, 
                "is_initial": cls.is_initial}
    
    @classmethod
    def build_prompt(cls,
                     question: str, 
                     previous_cot: Optional[str] = None) -> str:
        """
        Dynamically build the prompt with strategy prompt template.
        
        Args:
            question (str): The question to be answered.
            strategy_description (str): Description of the strategy.
            previous_cot (Optional[str]): Previous chain of thought. Required if not initial.

        Returns:
            str: Formatted prompt string
            
        Raises:
            ValueError: If not all required inputs are provided
        """
        prompt = StrategyPromptTemplate.create_header(question=question)
        
        if not cls.is_initial and previous_cot is None:
            raise ValueError("Previous CoT is required for non-initial strategies.")
        
        if cls.is_initial:
            prompt+= StrategyPromptTemplate.create_initial_instruction()
        else:
            prompt += StrategyPromptTemplate.create_previous_reasoning(previous_cot=previous_cot)
            prompt += StrategyPromptTemplate.create_new_instruction(strategy_description=cls.description)
        
        prompt += StrategyPromptTemplate.create_response_requirements()
        prompt += StrategyPromptTemplate.create_json_format()
        return prompt

@dataclass(frozen=True)
class InitializeCoT(Strategy):
    "Required strategy that kicks off CoT generation"
    name: ClassVar[str] = "intialize"
    description: ClassVar[str] = "initialize the chain of thought"
    is_initial: ClassVar[bool] = True
    
@dataclass(frozen=True)
class Backtrack(Strategy):
    "Refine the reasoning using backtracking to revisit earlier points of reasoning."
    name: ClassVar[str] = "backtrack"
    description: ClassVar[str] = prompts.backtrack_strategy_prompt
    is_initial: ClassVar[bool] = False