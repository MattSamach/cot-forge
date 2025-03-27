# Reasoning Strategies

Reasoning strategies in CoT Forge represent individual approaches for generating reasoning steps at specific nodes in a reasoning path. These strategies guide how the LLM should reason at each step of the process.

## Core Concept

A strategy is a specific instruction that tells the LLM how to approach reasoning at a particular point. For example, one strategy might instruct the model to validate its previous conclusions, while another might direct it to explore alternative approaches.

## Available Strategies

CoT Forge comes with several built-in strategies:

### Initial Strategy

- **InitializeCoT**: The required first strategy that begins the reasoning process
  ```python
  # Always used as the first step in any reasoning chain
  ```

### Continuation Strategies

- **Backtrack**: Revisits earlier reasoning points to refine the approach
  ```python
  # Useful when the reasoning path has reached a dead end
  # Requires minimum_depth=2
  ```

- **ExploreNewPaths**: Finds alternative approaches to solving the problem
  ```python
  # Useful for discovering different solution methods
  ```

- **Correction**: Makes precise corrections to address identified flaws
  ```python
  # Useful when specific errors are detected
  ```

- **Validation**: Conducts thorough validation to ensure reasoning validity
  ```python
  # Helps verify intermediate conclusions
  ```

### Additional Strategies

- **PerspectiveShift**: Adopts multiple perspectives to analyze the problem
- **AnalogicalReasoning**: Uses analogies to map the problem to better-understood domains
- **Decomposition**: Breaks down problems into smaller, manageable sub-problems
- **Counterfactual**: Explores what-if scenarios under different assumptions
- **FirstPrinciples**: Breaks down problems to fundamental principles

## Strategy Interface

Each strategy is defined as a dataclass that inherits from the `Strategy` base class:

```python
from cot_forge.reasoning.strategies import Strategy
from dataclasses import dataclass
from typing import ClassVar

@dataclass(frozen=True)
class MyCustomStrategy(Strategy):
    """Strategy description goes here."""
    name: ClassVar[str] = "my_strategy_name"
    description: ClassVar[str] = "Detailed instruction for the LLM"
    is_initial: ClassVar[bool] = False  # True only for starting strategies
    minimum_depth: ClassVar[int] = 0    # Minimum depth required
```

## Strategy Registry

The `StrategyRegistry` manages available strategies and provides methods to register, retrieve, and serialize them:

```python
from cot_forge.reasoning.strategies import StrategyRegistry, default_strategy_registry

# Using built-in registry
builder = CoTBuilder(
    search_llm=llm,
    search=search,
    strategy_reg=default_strategy_registry
)

# Creating custom registry
custom_registry = StrategyRegistry()
custom_registry.register_strategy(MyCustomStrategy)
```

## Dynamic Strategy Creation

You can create strategies dynamically without defining new classes:

```python
# Create a new strategy
my_strategy = Strategy.create_strategy(
    name="novel_approach",
    description="Try a completely novel approach to the problem",
    is_initial=False,
    minimum_depth=1
)

# Create and register in one step
custom_registry.create_and_register(
    name="novel_approach",
    description="Try a completely novel approach to the problem",
    is_initial=False,
    minimum_depth=1
)
```

## Strategy Selection

Strategies are selected during the search process using a strategy selector:

```python
from cot_forge.reasoning.strategies.strategy_selector import RandomStrategySelector

# Choose strategies randomly
selector = RandomStrategySelector()
strategies, info = selector.select(
    registry=default_strategy_registry,
    depth=current_depth,
    num_strategies=1
)
```

### Scored Selection

For more sophisticated selection, use `ScoredStrategySelector` to choose strategies based on their effectiveness:

```python
from cot_forge.reasoning.strategies.strategy_selector import ScoredStrategySelector
from cot_forge.reasoning.scorers import ProbabilityFinalAnswerScorer

# Create a selector that uses a scorer to evaluate strategies
selector = ScoredStrategySelector()
scorer = ProbabilityFinalAnswerScorer(llm_provider=llm)

# Select strategies by scoring their performance
strategies, info = selector.select(
    search_llm=llm,
    registry=registry,
    depth=current_depth,
    question=question,
    ground_truth_answer=ground_truth,
    scorer=scorer,
    node=current_node
)
```

## Prompt Generation

Each strategy generates prompts for the LLM using modular components:

```python
# Generate a prompt for a strategy
prompt = strategy.build_prompt(
    question="What is the capital of France?",
    previous_cot="Previous reasoning steps..." # Not needed for initial strategies
)
```

The prompt structure includes:
1. Question context
2. Previous reasoning (for non-initial strategies)
3. Strategy-specific instruction
4. Response format requirements

## Usage in Search Algorithms

Strategies are typically used by search algorithms to generate reasoning nodes:

```python
# Inside a search algorithm
strategy = registry.get_strategy("validation")
prompt = strategy.build_prompt(question, node.get_full_cot())
response = llm_provider.generate(prompt)
# Process response into a new reasoning node
```

## Serialization

Strategies and the strategy registry can be serialized for persistence:

```python
# Serialize the registry
serialized = registry.serialize()

# Deserialize later
restored_registry = StrategyRegistry.deserialize(serialized)
```