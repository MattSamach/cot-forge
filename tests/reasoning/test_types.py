import unittest
from unittest.mock import MagicMock
import json

from cot_forge.reasoning.types import ReasoningNode, SearchResult
from cot_forge.reasoning.strategies import Strategy, StrategyRegistry

class TestSerializationDeserialization(unittest.TestCase):
    
    def setUp(self):
        # Create a simple strategy registry with mock strategies
        self.strategy_registry = StrategyRegistry()
        
        # Create a simple mock strategy
        self.mock_strategy = MagicMock(spec=Strategy)
        self.mock_strategy.name = "test_strategy"
        
        # Register the mock strategy
        self.strategy_registry._strategies = {"test_strategy": self.mock_strategy}
        
        # Create a simple tree of reasoning nodes
        self.root_node = ReasoningNode(
            strategy=self.mock_strategy,
            prompt="Initial prompt",
            response="Initial response",
            cot=[{"thought": "Initial thought"}],
            is_final=False,
            success=False
        )
        
        self.child_node1 = ReasoningNode(
            strategy=self.mock_strategy,
            prompt="Child 1 prompt",
            response="Child 1 response",
            cot=[{"thought": "Child 1 thought"}],
            parent=self.root_node,
            is_final=False,
            success=False
        )
        
        self.child_node2 = ReasoningNode(
            strategy=self.mock_strategy,
            prompt="Child 2 prompt",
            response="Child 2 response",
            cot=[{"thought": "Child 2 thought"}],
            parent=self.root_node,
            is_final=True,
            success=True
        )
        
        # Link nodes together
        self.root_node.add_child(self.child_node1)
        self.root_node.add_child(self.child_node2)
        
        # Create a SearchResult with these nodes
        self.search_result = SearchResult(
            question="Test question?",
            ground_truth_answer="Test answer",
            terminal_nodes=[self.child_node1, self.child_node2],
            success=True,
            metadata={"test_key": "test_value"}
        )

    def test_serialization_basic(self):
        """Test that serialization produces a valid dictionary with expected keys"""
        serialized = self.search_result.serialize()
        
        # Verify the serialized dict has all expected top-level keys
        expected_keys = {"adjacency_list", "node_map", "question", "ground_truth_answer", 
                         "success", "metadata"}
        self.assertEqual(set(serialized.keys()), expected_keys)
        
        # Verify the basic fields are correct
        self.assertEqual(serialized["question"], "Test question?")
        self.assertEqual(serialized["ground_truth_answer"], "Test answer")
        self.assertEqual(serialized["success"], True)
        self.assertEqual(serialized["metadata"], {"test_key": "test_value"})
        
        # Verify we have the correct number of nodes
        self.assertEqual(len(serialized["node_map"]), 3)
        
        print("Adjacency List:", serialized["adjacency_list"])
        # Verify the adjacency list structure
        self.assertEqual(len(serialized["adjacency_list"]), 3)
        
        # Verify serialization can be converted to JSON
        try:
            json_str = json.dumps(serialized)
            self.assertIsInstance(json_str, str)
        except Exception as e:
            self.fail(f"JSON serialization failed: {e}")

    def test_round_trip_serialization(self):
        """Test that serialization followed by deserialization preserves structure"""
        serialized = self.search_result.serialize()
        
        # Mock the get_strategy method to return our mock strategy
        self.strategy_registry.get_strategy = MagicMock(return_value=self.mock_strategy)
        
        # Deserialize back to a SearchResult
        deserialized = SearchResult.deserialize(serialized, self.strategy_registry)
        
        # Verify the basic fields are preserved
        self.assertEqual(deserialized.question, self.search_result.question)
        self.assertEqual(deserialized.ground_truth_answer, self.search_result.ground_truth_answer)
        self.assertEqual(deserialized.success, self.search_result.success)
        self.assertEqual(deserialized.metadata, self.search_result.metadata)
        
        # Verify we have the correct number of terminal nodes
        self.assertEqual(len(deserialized.terminal_nodes), len(self.search_result.terminal_nodes))
        
        # Get a terminal node for further inspection
        terminal_node = deserialized.terminal_nodes[0]
        
        # Verify node structure is preserved by checking node hierarchy
        parent = terminal_node.parent
        self.assertIsNotNone(parent, "Terminal node should have a parent")
        
        # Check that the parent has the expected number of children
        self.assertEqual(len(parent.children), 2, "Root node should have 2 children")

if __name__ == '__main__':
    unittest.main()