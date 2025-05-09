from dataclasses import dataclass
from typing import ClassVar

import pytest

from cot_forge.reasoning.strategies import (
    Backtrack,
    InitializeCoT,
    Strategy,
    StrategyRegistry,
    backtrack_strategy_prompt,
    default_strategy_registry,
    initialize_cot_prompt,
)


class TestStrategy:
  def test_create_strategy(self):
    # Test the factory method for strategy creation
    test_strategy = Strategy.create_strategy(
        name="test_strategy",
        description="A test strategy",
        is_initial=True
    )

    assert test_strategy.name == "test_strategy"
    assert test_strategy.description == "A test strategy"
    assert test_strategy.is_initial

  def test_get_metadata(self):
    # Test metadata retrieval
    metadata = InitializeCoT.get_metadata()

    assert metadata["name"] == "initialize"
    assert metadata["description"] == initialize_cot_prompt
    assert metadata["is_initial"]

  def test_build_prompt_initial_strategy(self):
    # Test prompt building for initial strategy
    prompt = InitializeCoT.build_prompt(question="What is 2+2?")

    assert "<question>" in prompt
    assert "What is 2+2?" in prompt
    # Initial doesn't include strategy description
    assert "initialize the chain of thought" not in prompt
    assert "<response_requirements>" in prompt

  def test_build_prompt_non_initial_strategy(self):
    # Test prompt building for non-initial strategy
    previous_cot = "I think the answer is 4."
    prompt = Backtrack.build_prompt(
        question="What is 2+2?",
        previous_cot=previous_cot
    )

    assert "<question>" in prompt
    assert "What is 2+2?" in prompt
    assert "<previous_reasoning>" in prompt
    assert "I think the answer is 4." in prompt
    assert backtrack_strategy_prompt in prompt  # Strategy description included
    assert "<response_requirements>" in prompt

  def test_build_prompt_missing_previous_cot(self):
    # Test validation of required previous_cot for non-initial strategies
    with pytest.raises(ValueError, match="Previous CoT is required"):
      Backtrack.build_prompt(question="What is 2+2?")


class TestStrategyRegistry:
  def test_init_empty(self):
    # Test creating an empty registry
    registry = StrategyRegistry()
    assert len(registry._strategies) == 0

  def test_init_with_strategies(self):
    # Test creating a registry with initial strategies
    registry = StrategyRegistry([InitializeCoT, Backtrack])
    assert len(registry._strategies) == 2
    assert registry.get_strategy("initialize") == InitializeCoT
    assert registry.get_strategy("backtrack") == Backtrack

  def test_register_decorator(self):
    # Test registering a strategy using the decorator pattern
    registry = StrategyRegistry()

    @registry.register
    @dataclass(frozen=True)
    class TestStrategy(Strategy):
      name: ClassVar[str] = "test_decorator"
      description: ClassVar[str] = "A test strategy"
      is_initial: ClassVar[bool] = False

    assert registry.get_strategy("test_decorator") == TestStrategy

  def test_create_and_register(self):
    # Test creating and registering a strategy in one step
    registry = StrategyRegistry()
    new_strategy = registry.create_and_register(
        name="new_test_strategy",
        description="Another test strategy",
        is_initial=True
    )

    assert registry.get_strategy("new_test_strategy") == new_strategy
    assert new_strategy.is_initial

  def test_get_strategy(self):
    # Test getting strategies by name
    registry = StrategyRegistry([InitializeCoT])

    assert registry.get_strategy("initialize") == InitializeCoT
    assert registry.get_strategy("non_existent") is None

  def test_list_strategies(self):
    # Test listing all registered strategies
    registry = StrategyRegistry([InitializeCoT, Backtrack])

    strategy_names = registry.list_strategies()
    assert len(strategy_names) == 2
    assert "initialize" in strategy_names
    assert "backtrack" in strategy_names

  def test_get_all_strategies_metadata(self):
    # Test getting metadata for all strategies
    registry = StrategyRegistry([InitializeCoT, Backtrack])

    metadata = registry.get_all_strategies_metadata()
    assert len(metadata) == 2
    assert metadata["initialize"]["is_initial"]
    assert not metadata["backtrack"]["is_initial"]

  def test_remove_strategy(self):
    # Test removing a strategy
    registry = StrategyRegistry([InitializeCoT, Backtrack])

    registry.remove_strategy("backtrack")
    assert len(registry._strategies) == 1
    assert registry.get_strategy("backtrack") is None

  def test_remove_nonexistent_strategy(self):
    # Test removing a non-existent strategy raises error
    registry = StrategyRegistry([InitializeCoT])

    with pytest.raises(ValueError, match="not found in registry"):
      registry.remove_strategy("non_existent")


class TestDefaultRegistry:
  def test_default_registry_initialization(self):
    # Test that default registry is properly initialized with default strategies
    assert default_strategy_registry.get_strategy("initialize") is not None
    assert default_strategy_registry.get_strategy("backtrack") is not None
    assert default_strategy_registry.get_strategy(
        "explore_new_paths") is not None
    assert default_strategy_registry.get_strategy("correction") is not None
    assert default_strategy_registry.get_strategy("validation") is not None

  def test_default_registry_extensibility(self):
    # Test that default registry can be extended
    original_count = len(default_strategy_registry.list_strategies())

    # Add a new strategy
    default_strategy_registry.create_and_register(
        name="custom_test_strategy",
        description="A custom test strategy",
        is_initial=False
    )

    # Verify it was added
    assert len(default_strategy_registry.list_strategies()
               ) == original_count + 1
    assert default_strategy_registry.get_strategy(
        "custom_test_strategy") is not None

    # Clean up (remove the strategy we added)
    default_strategy_registry.remove_strategy("custom_test_strategy")


class TestSerialization:
  def test_strategy_to_dict(self):
    # Test serialization of a single strategy
    test_strategy = Strategy.create_strategy(
        name="serialization_test",
        description="Testing serialization",
        is_initial=True,
        minimum_depth=2
    )

    serialized = test_strategy.to_dict()

    assert serialized["name"] == "serialization_test"
    assert serialized["description"] == "Testing serialization"
    assert serialized["is_initial"] is True
    assert serialized["minimum_depth"] == 2

  def test_registry_serialization(self):
    # Test serialization of a registry
    registry = StrategyRegistry()
    registry.create_and_register(
        name="strategy1",
        description="First test strategy",
        is_initial=True
    )
    registry.create_and_register(
        name="strategy2",
        description="Second test strategy",
        is_initial=False,
        minimum_depth=3
    )

    serialized = registry.serialize()

    assert "strategies" in serialized
    assert len(serialized["strategies"]) == 2
    assert "strategy1" in serialized["strategies"]
    assert "strategy2" in serialized["strategies"]
    assert serialized["strategies"]["strategy1"]["is_initial"] is True
    assert serialized["strategies"]["strategy2"]["minimum_depth"] == 3

  def test_registry_deserialization(self):
    # Test deserialization of a registry
    original_registry = StrategyRegistry()
    original_registry.create_and_register(
        name="test_strat1",
        description="First test strategy",
        is_initial=True
    )
    original_registry.create_and_register(
        name="test_strat2",
        description="Second test strategy",
        is_initial=False,
        minimum_depth=4
    )

    # Serialize and then deserialize
    serialized_data = original_registry.serialize()
    deserialized_registry = StrategyRegistry.deserialize(serialized_data)

    # Verify the deserialized registry has the same strategies
    assert len(deserialized_registry.list_strategies()) == 2
    assert "test_strat1" in deserialized_registry.list_strategies()
    assert "test_strat2" in deserialized_registry.list_strategies()

    # Get strategies and verify their properties
    strat1 = deserialized_registry.get_strategy("test_strat1")
    strat2 = deserialized_registry.get_strategy("test_strat2")

    assert strat1.is_initial is True
    assert strat1.description == "First test strategy"
    assert strat2.is_initial is False
    assert strat2.minimum_depth == 4

  def test_empty_registry_serialization(self):
    # Test serialization of an empty registry
    empty_registry = StrategyRegistry()
    serialized = empty_registry.serialize()

    assert "strategies" in serialized
    assert len(serialized["strategies"]) == 0

    # Test round-trip
    deserialized = StrategyRegistry.deserialize(serialized)
    assert len(deserialized.list_strategies()) == 0

  def test_default_registry_serialization(self):
    # Test that we can serialize and deserialize the default registry
    serialized = default_strategy_registry.serialize()
    deserialized = StrategyRegistry.deserialize(serialized)

    # The deserialized registry should have the same strategies
    original_strategies = set(default_strategy_registry.list_strategies())
    deserialized_strategies = set(deserialized.list_strategies())

    assert original_strategies == deserialized_strategies

    # Verify a specific strategy was preserved correctly
    original_init = default_strategy_registry.get_strategy("initialize")
    deserialized_init = deserialized.get_strategy("initialize")

    assert original_init.name == deserialized_init.name
    assert original_init.is_initial == deserialized_init.is_initial
