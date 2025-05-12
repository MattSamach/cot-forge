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

### Key Features

1. **Strategy Class**: A base class for defining reasoning strategies with 
    attributes like `name`, `description`, `is_initial`, and `minimum_depth`.

2. **Dynamic Strategy Creation**: Use `Strategy.create_strategy()` to 
    dynamically create new strategies at runtime.

3. **Strategy Registry**: The `StrategyRegistry` class allows for 
    registering, retrieving, and managing strategies.

4. **Convenient Creation and Registration**: Use `StrategyRegistry.create_and_register()` 
    to create and register a strategy in one step.

Example usage of `create_and_register()`:

```python
from cot_forge.reasoning.strategies import default_strategy_registry

# Create and register a new strategy
default_strategy_registry.create_and_register(
    name="explore_alternatives",
    description="Explore alternative approaches to the problem",
    is_initial=False,
    minimum_depth=2
)
```
"""

from dataclasses import dataclass
from typing import Any, ClassVar

from . import prompts
from .prompts import StrategyPromptTemplate


@dataclass(frozen=True)
class Strategy:
  """
  Base class for defining reasoning strategies.

  This class provides a blueprint for creating strategies that can be used 
  in a chain of thought (CoT) reasoning process. Each strategy has attributes 
  like `name`, `description`, and `is_initial` to define its behavior.

  Attributes:
      name (str): The unique name of the strategy.
      description (str): A brief description of the strategy's purpose.
      is_initial (bool): Whether this strategy is the starting point in the CoT process.
      minimum_depth (int): The minimum depth required for this strategy to be applied.
  """
  name: ClassVar[str]
  description: ClassVar[str]
  is_initial: ClassVar[bool]
  minimum_depth: ClassVar[int] = 0

  @classmethod
  def create_strategy(
      cls,
      name: str,
      description: str,
      is_initial: bool = False,
      minimum_depth: int = 0
  ) -> type['Strategy']:
    """
    Factory method to dynamically create custom `Strategy` subclasses.

    This method allows you to define new strategies at runtime by specifying 
    their attributes. The resulting subclass can be used like any other 
    `Strategy` class.

    Args:
        name (str): The unique name of the strategy.
        description (str): A brief description of the strategy's purpose.
        is_initial (bool, optional): Whether this strategy is the starting point 
            in the CoT process. Defaults to False.
        minimum_depth (int, optional): The minimum depth required for this 
            strategy to be applied. Defaults to 0.

    Returns:
        type[Strategy]: A dynamically created subclass of `Strategy`.

    Example:
        ```
        ExploreAlternatives = Strategy.create_strategy(
            name="explore_alternatives",
            description="Explore alternative approaches to the problem",
            is_initial=False,
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
    Build a dynamic prompt using the strategy's prompt template.

    This method constructs a prompt based on the strategy's attributes, 
    the provided question, and optionally the previous chain of thought (CoT). 
    It ensures that all required inputs are provided before generating the prompt.

    Args:
        question (str): The question to be answered.
        previous_cot (str | None, optional): The previous chain of thought. 
            Required if the strategy is not an initial strategy.

    Returns:
        str: A formatted prompt string ready for use.

    Raises:
        ValueError: If `previous_cot` is not provided for non-initial strategies.
    """
    prompt = StrategyPromptTemplate.create_header(question=question)

    if not cls.is_initial and previous_cot is None:
      raise ValueError("Previous CoT is required for non-initial strategies.")

    if cls.is_initial:
      prompt += StrategyPromptTemplate.create_initial_instruction()
    else:
      prompt += StrategyPromptTemplate.create_previous_reasoning(
          previous_cot=previous_cot)
      prompt += StrategyPromptTemplate.create_new_instruction(
          strategy_description=cls.description)

    prompt += StrategyPromptTemplate.create_response_requirements()
    prompt += StrategyPromptTemplate.create_json_format()
    return prompt

  @classmethod
  def to_dict(cls) -> dict[str, Any]:
    """Convert the strategy to a dictionary representation.

    Returns:
        dict: A dictionary containing the strategy's metadata.
    """
    return {
        "name": cls.name,
        "description": cls.description,
        "is_initial": cls.is_initial,
        "minimum_depth": cls.minimum_depth
    }

  def __str__(self) -> str:
    """Return a string representation of the strategy.

    Returns:
        str: A string with the strategy name and description.
    """
    return f"Strategy(name='{self.__class__.name}', description='{self.__class__.description[:50]}...')"

  def __repr__(self) -> str:
    """Return a detailed representation of the strategy.

    Returns:
        str: A detailed string representation including all class attributes.
    """
    return (f"Strategy(name='{self.__class__.name}', "
            f"description='{self.__class__.description[:50]}...', "
            f"is_initial={self.__class__.is_initial}, "
            f"minimum_depth={self.__class__.minimum_depth})")


class StrategyRegistry:
  """
  A registry for managing reasoning strategies.

  The `StrategyRegistry` class provides methods to register, retrieve, 
  and manage strategies. It supports both manual registration of strategy 
  classes and dynamic creation and registration of new strategies.

  Attributes:
      _strategies (dict): A dictionary mapping strategy names to their 
          corresponding `Strategy` classes.
  """

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
    Register a `Strategy` class with the registry.

    This method adds a `Strategy` subclass to the registry, making it 
    available for retrieval by name. It can also be used as a decorator 
    to simplify the registration process.

    Args:
        strategy_class (type[Strategy]): The `Strategy` subclass to register.

    Returns:
        type[Strategy]: The registered `Strategy` subclass, enabling decorator usage.

    Example:
        ```
        @default_strategy_registry.register
        @dataclass(frozen=True)
        class MyCustomStrategy(Strategy):
            name: ClassVar[str] = "my_custom_strategy"
            description: ClassVar[str] = "A custom strategy."
            is_initial: ClassVar[bool] = False
        ```
    """
    self._strategies[strategy_class.name] = strategy_class
    return strategy_class

  def create_and_register(self,
                          name: str,
                          description: str,
                          is_initial: bool = False,
                          minimum_depth: int = 0
                          ) -> Strategy:
    """
    Create and register a new strategy in one step.

    This method combines the creation of a custom `Strategy` subclass 
    with its registration in the registry. It is a convenient way to 
    define and immediately make a strategy available for use.

    Args:
        name (str): The unique name of the strategy.
        description (str): A brief description of the strategy's purpose.
        is_initial (bool, optional): Whether this strategy is the starting 
            point in the CoT process. Defaults to False.
        minimum_depth (int, optional): The minimum depth required for this 
            strategy to be applied. Defaults to 0.

    Returns:
        Strategy: The dynamically created and registered `Strategy` subclass.

    Example:
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
    strategy = Strategy.create_strategy(
        name, description, is_initial, minimum_depth)
    self._strategies[name] = strategy
    return strategy

  def get_strategy(self, name: str) -> Strategy | None:
    """Get a strategy by name, or None if not found."""
    return self._strategies.get(name, None)

  def list_strategies(self) -> list[str]:
    """Return a list of all registered strategy names."""
    return list(self._strategies.keys())

  def get_all_strategies_metadata(self) -> dict:
    """Return metadata for all registered strategies."""
    return {name: strategy.get_metadata() for name, strategy in self._strategies.items()}

  def remove_strategy(self, name: str):
    """Remove a strategy from the registry."""
    if name in self._strategies:
      del self._strategies[name]
    else:
      raise ValueError(f"Strategy '{name}' not found in registry.")

  def serialize(self) -> dict[str, Any]:
    """
    Serialize the strategy registry to a dictionary representation.

    This method converts all registered strategies into a serializable format
    by creating a mapping of strategy names to their serialized representations.
    Each strategy is serialized using its `to_dict()` method, capturing its
    essential properties like name, description, and configuration.

    Returns:
        dict[str, Any]: A serializable dictionary containing all registered
                    strategies in the format {"strategies": {name: strategy_dict}}
    """
    return {
        "strategies": {name: strategy.to_dict() for name, strategy in self._strategies.items()}
    }

  @classmethod
  def deserialize(cls, data: dict[str, Any]) -> 'StrategyRegistry':
    """
    Reconstruct a StrategyRegistry from its serialized dictionary representation.

    This class method creates a new StrategyRegistry and populates it with strategies
    reconstructed from the serialized data. Each strategy is dynamically created using
    the create_and_register() method with the properties from its serialized form.

    Args:
        data (dict[str, Any]): The serialized registry dictionary containing a
                            "strategies" key mapping to serialized strategy data

    Returns:
        StrategyRegistry: A new registry instance populated with all the
                        deserialized strategies

    Raises:
        KeyError: If the serialized data is missing the "strategies" key
        ValueError: If any strategy's serialized data is invalid
    """
    registry = cls()
    if "strategies" not in data:
      raise KeyError("Serialized data must contain 'strategies' key.")

    for name, strategy_data in data["strategies"].items():
      try:
        registry.create_and_register(**strategy_data)
      except Exception as e:
        raise ValueError(
            f"Failed to deserialize strategy '{name}': {str(e)}") from e

    return registry

  def __str__(self) -> str:
    """Return a string representation of the registry.

    Returns:
        str: A string listing the number of registered strategies.
    """
    return f"StrategyRegistry(strategies={len(self._strategies)})"

  def __repr__(self) -> str:
    """Return a detailed representation of the registry.

    Returns:
        str: A detailed string representation including all registered strategy names.
    """
    strategy_names = ", ".join(self._strategies.keys())
    return f"StrategyRegistry(strategies=[{strategy_names}])"


@dataclass(frozen=True)
class InitializeCoT(Strategy):
  "Required strategy that kicks off CoT generation"
  name: ClassVar[str] = "initialize"
  description: ClassVar[str] = prompts.initialize_cot_prompt
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
