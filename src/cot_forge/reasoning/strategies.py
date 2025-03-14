"""
This file defines strategies that constitute individual links within
the chain of thought (CoT).

For typical usage, import from the reasoning module directly:

```python
from cot_forge.reasoning.strategies import Strategy, default_strategy_registry

# Create and register in one step
my_strategy = default_strategy_registry.create_and_register(
    name="my_custom_strategy",
    description="A custom strategy that does X",
    is_initial=False
)

# Or register an existing strategy class
@default_strategy_registry.register
@dataclass(frozen=True)
class MyCustomStrategy(Strategy):
    name: ClassVar[str] = "my_custom_strategy"
    description: ClassVar[str] = "A custom strategy that does X"
    is_initial: ClassVar[bool] = False
```
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
        minimum_depth (int): Minimum depth required for this strategy.
    """
    name: ClassVar[str]
    description: ClassVar[str]
    is_initial: ClassVar[bool]
    minimum_depth: ClassVar[int] = 0
    
    @classmethod
    def create_strategy(cls, name: str, description: str, is_initial: bool = False, minimum_depth: int = 0) -> type['Strategy']:
        """
        Factory method to create custom Strategy subclasses.
        
        Example usage to create a new strategy at runtime:
        ```
        ExploreAlternatives = Strategy.create_strategy(
            name="explore_alternatives",
            description="Explore alternative approaches to the problem",
            is_initial=False
            minimum_depth=2
        )
        
        print(ExploreAlternatives.build_prompt("What is the capital of France?"))
        ```
        """
        return type(name, (cls,), {
            "name": name,
            "description": description,
            "is_initial": is_initial,
            "minimum_depth": minimum_depth,
            "__doc__": description
        })
    
    @classmethod
    def get_metadata(cls) -> dict[str, Any]:
        return {"name": cls.name, 
                "description": cls.description, 
                "is_initial": cls.is_initial,
                "minimum_depth": cls.minimum_depth}
    
    @classmethod
    def build_prompt(cls,
                     question: str,
                     previous_cot: str | None = None) -> str:
        """
        Dynamically build the prompt with strategy prompt template.
        
        Args:
            question: The question to be answered.
            strategy_description: Description of the strategy.
            previous_cot: Previous chain of thought. Required if not initial.

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
    
class StrategyRegistry:
    """ Registry for strategies. Allows quick lookup and registration of strategies. """
    
    def __init__(self, strategies: list[Strategy] | None = None):
        """ Initialize an empty strategy registry. 
        
        Args:
            strategies: List of initial strategies to register. Defaults to None.
        """
        if strategies is None:
            self._strategies = {}
        else:
            self._strategies = {strategy.name: strategy for strategy in strategies}
    
    def register(self, strategy_class: type[Strategy]) -> type[Strategy]:
        """
        Register a Strategy class with this registry.
        Can be used as a decorator.
        
        Args:
            strategy_class: A Strategy subclass to register
            
        Returns:
            The strategy_class (enabling decorator usage)
        """
        self._strategies[strategy_class.name] = strategy_class
        return strategy_class
    
    def create_and_register(self, name: str, 
                            description: str, 
                            is_initial: bool = False, 
                            minimum_depth: int = 0
                            ) -> Strategy:
        """
        Create and register a new strategy at the same time.

        Example usage:
        ```
        registry = StrategyRegistry()
        registry.create_and_register(
            name="explore_alternatives",
            description="Explore alternative approaches to the problem",
            is_initial=False,
            minimum_depth=2
        )
        ```
        """
        strategy = Strategy.create_strategy(name, description, is_initial, minimum_depth)
        self._strategies[name] = strategy
        return strategy
    
    def get_strategy(self, name: str) -> Optional[Strategy]:
        """Get a strategy by name, or None if not found."""
        return self._strategies.get(name, None)
    
    def list_strategies(self):
        """Return a list of all registered strategy names."""
        return list(self._strategies.keys())
    
    def get_all_strategies_metadata(self):
        """Return metadata for all registered strategies."""
        return {name: strategy.get_metadata() for name, strategy in self._strategies.items()}
    
    def remove_strategy(self, name: str):
        """Remove a strategy from the registry."""
        if name in self._strategies:
            del self._strategies[name]
        else:
            raise ValueError(f"Strategy '{name}' not found in registry.")

@dataclass(frozen=True)
class InitializeCoT(Strategy):
    "Required strategy that kicks off CoT generation"
    name: ClassVar[str] = "initialize"
    description: ClassVar[str] = "initialize the chain of thought"
    is_initial: ClassVar[bool] = True
    
@dataclass(frozen=True)
class Backtrack(Strategy):
    "Refine the reasoning using backtracking to revisit earlier points of reasoning."
    name: ClassVar[str] = "backtrack"
    description: ClassVar[str] = prompts.backtrack_strategy_prompt
    is_initial: ClassVar[bool] = False
    minimum_depth: ClassVar[int] = 2
    
@dataclass(frozen=True)
class ExploreNewPaths(Strategy):
    "Refine the reasoning by exploring new approaches to solving this problem."
    name: ClassVar[str] = "explore_new_paths"
    description: ClassVar[str] = prompts.explore_new_paths_strategy_prompt
    is_initial: ClassVar[bool] = False
    
@dataclass(frozen=True)
class Correction(Strategy):
    "Refine the reasoning by making precise corrections to address prior flaws."
    name: ClassVar[str] = "correction"
    description: ClassVar[str] = prompts.correction_strategy_prompt
    is_initial: ClassVar[bool] = False
    
@dataclass(frozen=True)
class Validation(Strategy):
    "Refine the reasoning by conducting a thorough validation process to ensure validity."
    name: ClassVar[str] = "validation"
    description: ClassVar[str] = prompts.validation_strategy_prompt
    is_initial: ClassVar[bool] = False

# Default strategy registry with common reasoning strategies for use in search
default_strategies = [
    InitializeCoT,
    Backtrack,
    ExploreNewPaths,
    Correction,
    Validation
]

# Default strategy registry for export
default_strategy_registry = StrategyRegistry(strategies=default_strategies)

# Additional strategies
@dataclass(frozen=True)
class PerspectiveShift(Strategy):
    """Refine the reasoning by adopting multiple different
    perspectives to analyze the problem from various angles."""
    name: ClassVar[str] = "perspective_shift"
    description: ClassVar[str] = prompts.perspective_shift_strategy_prompt
    is_initial: ClassVar[bool] = False

@dataclass(frozen=True)
class AnalogicalReasoning(Strategy):
    """Refine the reasoning by using analogies or metaphors to map the
    problem to a better-understood domain, then transfer insights back to the original problem."""
    name: ClassVar[str] = "analogical_reasoning"
    description: ClassVar[str] = prompts.analogical_reasoning_prompt
    is_initial: ClassVar[bool] = False
    
@dataclass(frozen=True)
class Decomposition(Strategy):
    """Refine the reasoning by breaking down the problem into smaller, 
    more manageable sub-problems, solving each one systematically."""
    name: ClassVar[str] = "decomposition"
    description: ClassVar[str] = prompts.decomposition_strategy_prompt
    is_initial: ClassVar[bool] = False

@dataclass(frozen=True)
class Counterfactual(Strategy):
    """Refine the reasoning by exploring what-if scenarios and considering
    how outcomes might change under different conditions or assumptions."""
    name: ClassVar[str] = "counterfactual"
    description: ClassVar[str] = prompts.counterfactual_strategy_prompt
    is_initial: ClassVar[bool] = False
    
@dataclass(frozen=True)
class FirstPrinciples(Strategy):
    """Refine the reasoning by breaking down the problem into its
    fundamental principles and building a solution from the ground up."""
    name: ClassVar[str] = "first_principles"
    description: ClassVar[str] = prompts.first_principles_strategy_prompt
    is_initial: ClassVar[bool] = False