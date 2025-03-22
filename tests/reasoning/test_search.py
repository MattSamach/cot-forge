import unittest
from unittest.mock import MagicMock, patch

from cot_forge.llm import LLMProvider
from cot_forge.reasoning import NaiveLinearSearch, SimpleBeamSearch
from cot_forge.reasoning.scorers import BaseScorer
from cot_forge.reasoning.types import ReasoningNode
from cot_forge.reasoning.verifiers import LLMJudgeVerifier


class TestNaiveLinearSearch(unittest.TestCase):
    """Test that the naive linear search algorithm works as expected."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.llm_provider = MagicMock(spec=LLMProvider)
        self.verifier = MagicMock(spec=LLMJudgeVerifier)
        self.search = NaiveLinearSearch()
    
    def test_verification_success(self):
        """Test that the search algorithm returns a successful result."""
        question = "What is 2 + 2?"
        ground_truth_answer = "4"

        # Set up the mock response
        self.llm_provider.generate.return_value = """{
            "CoT": [
                {
                    "action": "Inner Thinking",
                    "title": "Add Numbers",
                    "content": "I believe the answer is 4 because 2 + 2 = 4."
                },
                {
                    "action": "Final Conclusion",
                    "content": "The answer is 4."
                }
            ]
        }"""
        self.verifier.return_value = True, "The answer is correct."

        # Run the search algorithm
        result = self.search(
            question=question,
            ground_truth_answer=ground_truth_answer,
            search_llm=self.llm_provider,
            verifier=self.verifier
        )
                        
        # Verify LLM was called with correct parameters
        self.llm_provider.generate.assert_called_once()
        # Check the prompt contains the question
        self.assertIn(question, self.llm_provider.generate.call_args[0][0])
        
        # Verify the verifier was called with correct parameters
        self.verifier.assert_called_once()
                
        # Check result structure and contents
        self.assertTrue(result.success)
        self.assertEqual(result.get_successful_final_answers()[0], "The answer is 4.")
        self.assertEqual(len(result.terminal_nodes), 1)
        self.assertIn('depth', result.metadata)
        
    def test_verification_failure(self):
        """Test handling of verification failure."""
        question = "What is 2 + 2?"
        ground_truth_answer = "4"
        
        # Set up the mock response with an incorrect answer
        self.llm_provider.generate.return_value = """{
            "CoT": [
                {
                    "action": "Inner Thinking",
                    "title": "Add Numbers",
                    "content": "I believe the answer is 5 because 2 + 2 = 5."
                },
                {
                    "action": "Final Conclusion",
                    "content": "The answer is 5."
                },
                {
                    "action": "Verification",
                    "content": "The answer is correct because 2 + 2 = 5."
                }
            ]
        }"""
        self.verifier.return_value = False, "The answer is incorrect."

        # Run the search algorithm with default max_depth=1
        result = self.search(
            question=question,
            ground_truth_answer=ground_truth_answer,
            search_llm=self.llm_provider,
            verifier=self.verifier
        )
        
        # Check result indicates failure
        self.assertFalse(result.success)
        self.assertEqual(result.get_all_final_answers()[0], "The answer is 5.")
        self.assertEqual(len(result.terminal_nodes), 1)
        
    @patch('cot_forge.reasoning.verifiers.LLMJudgeVerifier')
    @patch('cot_forge.llm.LLMProvider')
    def test_multiple_attempts_success(self, mock_llm_provider, mock_verifier):
        """Test successful search after initial failure."""
        question = "What is 2 + 2?"
        ground_truth_answer = "4"
        
        # Set up mock responses - first incorrect, then correct
        mock_responses = [
            """{
                "CoT": [
                    {
                        "action": "Inner Thinking",
                        "title": "Add Numbers",
                        "content": "I believe the answer is 5 because 2 + 2 = 5."
                    },
                    {
                        "action": "Final Conclusion",
                        "content": "The answer is 5."
                    }
                ]
            }""",
            """{
                "CoT": [
                    {
                        "action": "Inner Thinking",
                        "title": "Add Numbers",
                        "content": "I believe the answer is 4 because 2 + 2 = 4."
                    },
                    {
                        "action": "Final Conclusion",
                        "content": "The answer is 4."
                    }
                ]
            }"""
        ]
        mock_llm_provider.generate.side_effect = mock_responses
        mock_verifier.side_effect = [
            (False, "The answer is incorrect."),
            (True, "The answer is correct.")
            ]

        # Run the search algorithm with max_depth=2
        result = self.search(
            question=question,
            ground_truth_answer=ground_truth_answer,
            search_llm=mock_llm_provider,
            verifier=mock_verifier,
            max_depth=2
        )
        
        # Check LLM was called twice
        self.assertEqual(mock_llm_provider.generate.call_count, 2)
        self.assertEqual(mock_verifier.call_count, 2)
        
        # Check result indicates success
        self.assertTrue(result.success)
        self.assertEqual(result.get_successful_final_answers()[0], "The answer is 4.")
        self.assertEqual(len(result.terminal_nodes), 1)
        self.assertEqual(result.metadata['depth'], 2)
        
    @patch('cot_forge.reasoning.verifiers.LLMJudgeVerifier')
    @patch('cot_forge.llm.LLMProvider')
    def test_max_depth_reached(self, mock_llm_provider, mock_verifier):
        """Test that search stops when max_depth is reached."""
        question = "What is 2 + 2?"
        ground_truth_answer = "4"
        
        # Set up mock to consistently return incorrect answers
        incorrect_response = """{
            "CoT": [
                {
                    "action": "Inner Thinking",
                    "title": "Add Numbers",
                    "content": "I believe the answer is 5 because 2 + 2 = 5."
                },
                {
                    "action": "Final Conclusion",
                    "content": "The answer is 5."
                }
            ]
        }"""
        mock_llm_provider.generate.return_value = incorrect_response
        # mock_llm_provider.generate.return_value = incorrect_response
        mock_verifier.return_value = False, "The answer is incorrect."

        # Run the search algorithm with max_depth=3
        self.search.max_depth = 3
        result = self.search(
            question = question,
            ground_truth_answer = ground_truth_answer,
            search_llm = mock_llm_provider,
            verifier = mock_verifier,
        )
        
        # Check LLM was called exactly 3 times
        self.assertEqual(mock_llm_provider.generate.call_count, 3)
        self.assertEqual(mock_verifier.call_count, 3)
        
        # Check result indicates failure
        self.assertFalse(result.success)
        self.assertEqual(len(result.terminal_nodes), 1)
        self.assertEqual(result.metadata['depth'], 3)
        
    @patch('cot_forge.reasoning.verifiers.LLMJudgeVerifier')
    @patch('cot_forge.llm.LLMProvider')
    def test_malformed_llm_response(self, mock_llm_provider, mock_verifier):
        """Test handling of malformed LLM response."""
        question = "What is 2 + 2?"
        ground_truth_answer = "4"
        
        # Set up mock to return malformed JSON
        mock_llm_provider.generate.return_value = """{ "CoT": [ this is not valid JSON """
        
        # Run the search algorithm
        result = self.search(
            question=question,
            ground_truth_answer=ground_truth_answer,
            verifier=mock_verifier,
            search_llm=mock_llm_provider,
        )
        
        # Check result indicates failure
        self.assertFalse(result.success)
        self.assertEqual(len(result.terminal_nodes), 0)
        self.assertEqual(len(result.get_successful_final_answers()), 0)
        
    
    @patch('cot_forge.reasoning.verifiers.LLMJudgeVerifier')
    @patch('cot_forge.llm.LLMProvider')
    def test_llm_exception(self, mock_llm_provider, mock_verifier):
        """Test handling of LLM exception."""
        question = "What is 2 + 2?"
        ground_truth_answer = "4"
        
        # Set up mock to raise an exception
        mock_llm_provider.generate.side_effect = Exception("API error")
        
        # Run the search algorithm
        result = self.search(
            question=question,
            ground_truth_answer=ground_truth_answer,
            verifier=mock_verifier,
            search_llm=mock_llm_provider
        )
                                
        # Check result indicates failure
        self.assertFalse(result.success)
        self.assertEqual(len(result.terminal_nodes), 0)
        self.assertEqual(len(result.get_successful_final_answers()), 0)
        self.assertIn('error', result.metadata['reason'])
        
    @patch('cot_forge.reasoning.verifiers.LLMJudgeVerifier')
    def test_missing_final_conclusion(self, mock_verifier):
        """Test handling of LLM response without Final Conclusion."""
        
        mock_verifier.return_value = (False, "Error: No Final Conclusion found in response")

        question = "What is 2 + 2?"
        ground_truth_answer = "4"
        
        # Set up the mock response without Final Conclusion
        self.llm_provider.generate.return_value = """{
            "CoT": [
                {
                    "action": "Inner Thinking",
                    "title": "Add Numbers",
                    "content": "I believe the answer is 4 because 2 + 2 = 4."
                },
                {
                    "action": "Verification",
                    "content": "The answer is correct because 2 + 2 = 4."
                }
            ]
        }"""
        
        max_depth = 5
        self.search.max_depth = max_depth
        # Run the search algorithm
        result = self.search(
            question=question,
            ground_truth_answer=ground_truth_answer,
            search_llm=self.llm_provider,
            verifier=mock_verifier,
            max_depth=max_depth
        )
    
        # Check result indicates issue with parsing
        self.assertFalse(result.success)
        all(
            node.metadata.get('warning') == 'missing_final_conclusion' 
            for node in result.terminal_nodes[0].get_full_node_chain()
        )
        self.assertEqual(len(result.get_successful_final_answers()),  0)
        # Chain should have gone to max depth
        assert result.metadata['depth'] == max_depth
        
            
class TestSimpleBeamSearch(unittest.TestCase):
    """Test that the simple beam search algorithm works as expected."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.search = MagicMock(spec=SimpleBeamSearch)
        self.llm_provider = MagicMock(spec=LLMProvider)
        self.verifier = MagicMock(spec=LLMJudgeVerifier)
        self.scorer = MagicMock(spec=BaseScorer)
        
    @patch('cot_forge.reasoning.strategies.ScoredStrategySelector')
    @patch('cot_forge.utils.parsing.extract_cot')
    @patch('cot_forge.reasoning.verifiers.BaseVerifier')
    @patch('cot_forge.llm.LLMProvider')
    def test_verification_success(self,
                                  mock_llm_provider,
                                  mock_verifier,
                                  mock_extract_cot,
                                  mock_strategy_selector_class):
        """Test that beam search correctly identifies a successful path."""
        question = "What is 2 + 2?"
        ground_truth_answer = "4"
        
        # Set up mock responses
        mock_llm_provider.generate.side_effect = [
            "Initial thinking",
            "Strategy 1 response",
            "Strategy 2 response",
            "Strategy 3 response",
        ]
        
        # Mock CoT extraction
        mock_extract_cot.side_effect = [
            [{"action": "Initial", "content": "Let me think"}],
            [{"action": "Final Conclusion", "content": "The answer is 4."}],
            [{"action": "Final Conclusion", "content": "The answer is 5."}],
            [{"action": "Final Conclusion", "content": "The answer is 4."}],
        ]
        
        # Create strategy mocks with proper mock methods
        strategy1 = MagicMock()
        strategy1.name = "Strategy1"
        strategy1.build_prompt = MagicMock(return_value="Prompt for Strategy1")
        strategy1.is_initial = False
        strategy1.minimum_depth = 1
        
        strategy2 = MagicMock()
        strategy2.name = "Strategy2"
        strategy2.build_prompt = MagicMock(return_value="Prompt for Strategy2")
        strategy2.is_initial = False
        strategy2.minimum_depth = 1
        
        strategy3 = MagicMock()
        strategy3.name = "Strategy3"
        strategy3.build_prompt = MagicMock(return_value="Prompt for Strategy3")
        strategy3.is_initial = False
        strategy3.minimum_depth = 1
        
        # Mock strategy registry
        mock_registry = MagicMock()
        mock_registry.get_strategy.side_effect = lambda x: {
            "Strategy1": strategy1,
            "Strategy2": strategy2,
            "Strategy3": strategy3
        }[x]
        mock_registry.list_strategies.return_value = ["Strategy1", "Strategy2", "Strategy3"]
        
        # Mock verifier to succeed for Strategy 1
        mock_verifier.side_effect = [
            (False, "Not yet correct"),  # Initial verification
            (True, "Correct!"),         # Strategy 1
            (False, "Incorrect"),       # Strategy 2
            (True, "Correct!"),         # Strategy 3
        ]
        
        # Mock scorer to give clear preferences
        mock_scorer = MagicMock(spec=BaseScorer)
        mock_scorer.return_value = {
            "Strategy1": 0.9,  # Highest score
            "Strategy2": 0.4,
            "Strategy3": 0.7,  # Second highest
        }
        
        # Create an instance of SimpleBeamSearch
        beam_search = SimpleBeamSearch(beam_width=2, branching_factor=3)
        
        # Create initial node for testing
        initial_node = ReasoningNode(
            strategy=MagicMock(),
            prompt="Initial prompt",
            response="Initial response",
            cot=[{"action": "Initial", "content": "Let me think"}],
            parent=None
        )
        
        # Mock initialize_cot to return our predefined initial node
        beam_search.initialize_cot = MagicMock(return_value=initial_node)
        
        # Create mock strategies dict
        mock_strategies_dict = {
            "Strategy1": {
                "score": 0.9,
                "strategy": strategy1,
                "cot": [{"action": "Final Conclusion", "content": "The answer is 4."}],
                "prompt": "Prompt for Strategy1",
                "response": "Strategy 1 response"
            },
            "Strategy2": {
                "score": 0.4,
                "strategy": strategy2,
                "cot": [{"action": "Final Conclusion", "content": "The answer is 5."}],
                "prompt": "Prompt for Strategy2",
                "response": "Strategy 2 response"
            },
            "Strategy3": {
                "score": 0.7,
                "strategy": strategy3,
                "cot": [{"action": "Final Conclusion", "content": "The answer is 4."}],
                "prompt": "Prompt for Strategy3",
                "response": "Strategy 3 response"
            }
        }
        
        # Mock the strategy_selector to return the expected tuple
        mock_selector = MagicMock()
        mock_strategy_selector_class.return_value = mock_selector
        mock_selector.select.return_value = (
            [strategy1, strategy3],  # Selected strategies (top 2 by score)
            {
                "strategies_dict": mock_strategies_dict,
                "scores": {
                    "Strategy1": 0.9,
                    "Strategy2": 0.4,
                    "Strategy3": 0.7,
                }
            }
        )
        
        # Replace the strategy_selector on beam_search with our mock
        beam_search.strategy_selector = mock_selector
        
        # Create mock beams for the first iteration
        beam1 = ReasoningNode(
            strategy=strategy1,
            prompt="Prompt for Strategy1",
            response="Strategy 1 response",
            cot=[{"action": "Final Conclusion", "content": "The answer is 4."}],
            parent=initial_node
        )
        beam1.success = True  # Mark as successful
        beam1.is_final = True
        
        beam2 = ReasoningNode(
            strategy=strategy3,
            prompt="Prompt for Strategy3",
            response="Strategy 3 response",
            cot=[{"action": "Final Conclusion", "content": "The answer is 4."}],
            parent=initial_node
        )
        beam2.success = True  # Mark as successful
        beam2.is_final = True
        
        # Mock initialize_beams to return our predefined beams
        beam_search.initialize_beams = MagicMock(return_value=[beam1, beam2])
        
        # Run the search
        result = beam_search._search(
            question=question,
            ground_truth_answer=ground_truth_answer,
            search_llm=mock_llm_provider,
            scorer=mock_scorer,
            verifier=mock_verifier,
            strategy_registry=mock_registry
        )
        
        # Check result indicates success
        self.assertTrue(result.success)
        self.assertIsNotNone(result.terminal_nodes)
        self.assertTrue(any(node.success for node in result.terminal_nodes))
        
        # Verify the highest scoring strategies were selected
        # We need to get the nodes at depth 1 which are children of initial_node
        nodes_at_depth_1 = []
        for terminal_node in result.terminal_nodes:
            # Traverse up to find the node at depth 1
            node = terminal_node
            while node.parent and node.parent != initial_node:
                node = node.parent
            if node.parent == initial_node:
                nodes_at_depth_1.append(node)
        
        # Get the strategies used at depth 1
        beam_strategies = [node.strategy for node in nodes_at_depth_1]
        
        # Verify the highest scoring strategies were selected
        self.assertTrue(strategy1 in beam_strategies)
        self.assertTrue(strategy3 in beam_strategies)
        self.assertTrue(strategy2 not in beam_strategies)
        
    @patch('cot_forge.reasoning.strategies.ScoredStrategySelector')
    @patch('cot_forge.utils.parsing.extract_cot')
    @patch('cot_forge.reasoning.verifiers.BaseVerifier')
    @patch('cot_forge.llm.LLMProvider')
    def test_verification_failure(self,
                                  mock_llm_provider,
                                  mock_verifier,
                                  mock_extract_cot,
                                  mock_strategy_selector_class):
        """Test handling of verification failure across all beams."""
        question = "What is 2 + 2?"
        ground_truth_answer = "4"
        
        # Create SimpleBeamSearch instance
        beam_search = SimpleBeamSearch(beam_width=2, branching_factor=2, max_depth=2)
        
        # Set up mock responses
        mock_llm_provider.generate.side_effect = [
            "Initial thinking about 2+2",
            "Strategy 1 response",
            "Strategy 2 response",
        ]
        
        # Mock CoT extraction
        mock_extract_cot.side_effect = [
            [{"action": "Initial", "content": "Let me think about 2+2"}],
            [{"action": "Final Conclusion", "content": "The answer is 5."}],
            [{"action": "Final Conclusion", "content": "The answer is 3."}],
        ]
        
        # Create strategy mocks with proper attributes
        strategy1 = MagicMock()
        strategy1.name = "Strategy1"
        strategy1.build_prompt = MagicMock(return_value="Prompt for Strategy1")
        strategy1.is_initial = False
        strategy1.minimum_depth = 1
        
        strategy2 = MagicMock()
        strategy2.name = "Strategy2"
        strategy2.build_prompt = MagicMock(return_value="Prompt for Strategy2")
        strategy2.is_initial = False
        strategy2.minimum_depth = 1
        
        # Mock strategy registry
        mock_registry = MagicMock()
        mock_registry.get_strategy.side_effect = lambda x: {
            "Strategy1": strategy1,
            "Strategy2": strategy2
        }[x]
        mock_registry.list_strategies.return_value = ["Strategy1", "Strategy2"]
        
        # Mock verifier to fail for all strategies
        mock_verifier.return_value = (False, "Incorrect answer")
        
        # Mock scorer
        mock_scorer = MagicMock(spec=BaseScorer)
        mock_scorer.return_value = {
            "Strategy1": 0.6,
            "Strategy2": 0.4,
        }
        
        # Create initial node
        initial_node = ReasoningNode(
            strategy=MagicMock(),
            prompt="Initial prompt",
            response="Initial response",
            cot=[{"action": "Initial", "content": "Let me think about 2+2"}],
            parent=None
        )
        
        # Mock initialize_cot to return our predefined initial node
        beam_search.initialize_cot = MagicMock(return_value=initial_node)
        
        # Create mock strategies dict
        mock_strategies_dict = {
            "Strategy1": {
                "score": 0.6,
                "strategy": strategy1,
                "cot": [{"action": "Final Conclusion", "content": "The answer is 5."}],
                "prompt": "Prompt for Strategy1",
                "response": "Strategy 1 response"
            },
            "Strategy2": {
                "score": 0.4,
                "strategy": strategy2,
                "cot": [{"action": "Final Conclusion", "content": "The answer is 3."}],
                "prompt": "Prompt for Strategy2",
                "response": "Strategy 2 response"
            }
        }
        
        # Mock the strategy_selector
        mock_selector = MagicMock()
        mock_strategy_selector_class.return_value = mock_selector
        mock_selector.select.return_value = (
            [strategy1, strategy2],
            {
                "strategies_dict": mock_strategies_dict,
                "scores": {
                    "Strategy1": 0.6,
                    "Strategy2": 0.4,
                }
            }
        )
        
        # Replace the strategy_selector on beam_search
        beam_search.strategy_selector = mock_selector
        
        # Create beams for the test - both failing verification
        beam1 = ReasoningNode(
            strategy=strategy1,
            prompt="Prompt for Strategy1",
            response="Strategy 1 response",
            cot=[{"action": "Final Conclusion", "content": "The answer is 5."}],
            parent=initial_node
        )
        beam1.success = False
        beam1.is_final = True
        
        beam2 = ReasoningNode(
            strategy=strategy2,
            prompt="Prompt for Strategy2",
            response="Strategy 2 response",
            cot=[{"action": "Final Conclusion", "content": "The answer is 3."}],
            parent=initial_node
        )
        beam2.success = False
        beam2.is_final = True
        
        # Mock initialize_beams to return our failed beams
        beam_search.initialize_beams = MagicMock(return_value=[beam1, beam2])
        
        # Run the search
        result = beam_search._search(
            question=question,
            ground_truth_answer=ground_truth_answer,
            search_llm=mock_llm_provider,
            scorer=mock_scorer,
            verifier=mock_verifier,
            strategy_registry=mock_registry
        )
        
        # Check result indicates failure
        self.assertFalse(result.success)
        self.assertIsNotNone(result.terminal_nodes)
        
    @patch('cot_forge.reasoning.strategies.ScoredStrategySelector')
    @patch('cot_forge.utils.parsing.extract_cot')
    @patch('cot_forge.reasoning.verifiers.BaseVerifier')
    @patch('cot_forge.llm.LLMProvider')
    def test_multiple_depth_success(self,
                                    mock_llm_provider,
                                    mock_verifier,
                                    mock_extract_cot,
                                    mock_strategy_selector_class):
        """Test successful search after multiple depths."""
        question = "What is 2 + 2?"
        ground_truth_answer = "4"
        
        # Create SimpleBeamSearch instance
        beam_search = SimpleBeamSearch(beam_width=2, branching_factor=2, max_depth=3)
        
        # Set up a complex mock response sequence
        mock_llm_provider.generate.side_effect = [
            "Initial thinking",
            "Beam 1 depth 1",
            "Beam 2 depth 1",
            "Beam 1 depth 2",
            "Beam 2 depth 2",
        ]
        
        # Mock CoT extraction - success at second depth
        mock_extract_cot.side_effect = [
            [{"action": "Initial", "content": "Let me think"}],  # Initial
            [{"action": "Step 1", "content": "First attempt"}],  # Beam 1 depth 1
            [{"action": "Step 1", "content": "Alternative approach"}],  # Beam 2 depth 1
            [{"action": "Final Conclusion", "content": "The answer is 4."}],  # Beam 1 depth 2
            [{"action": "Final Conclusion", "content": "The answer is 5."}],  # Beam 2 depth 2
        ]
        
        # Create strategy mocks
        strategy1 = MagicMock()
        strategy1.name = "Strategy1"
        strategy1.build_prompt = MagicMock(return_value="Prompt for Strategy1")
        strategy1.is_initial = False
        strategy1.minimum_depth = 1
        
        strategy2 = MagicMock()
        strategy2.name = "Strategy2"
        strategy2.build_prompt = MagicMock(return_value="Prompt for Strategy2")
        strategy2.is_initial = False
        strategy2.minimum_depth = 1
        
        # Mock strategy registry
        mock_registry = MagicMock()
        mock_registry.get_strategy.side_effect = lambda x: {
            "Strategy1": strategy1,
            "Strategy2": strategy2
        }[x]
        mock_registry.list_strategies.return_value = ["Strategy1", "Strategy2"]
        
        # Mock verifier to succeed only at depth 2 for first beam
        mock_verifier.side_effect = [
            (False, "Not yet"),  # Initial node verification
            (False, "Not yet"),  # Beam 1 depth 1
            (False, "Not yet"),  # Beam 2 depth 1
            (True, "Correct!"),  # Beam 1 depth 2
            (False, "Wrong"),    # Beam 2 depth 2
        ]
        
        # Mock scorer with consistent scoring
        mock_scorer = MagicMock(spec=BaseScorer)
        mock_scorer.side_effect = [
            {"Strategy1": 0.7, "Strategy2": 0.6},  # Depth 1 scoring
            {"Strategy1": 0.8, "Strategy2": 0.5},  # Depth 2 scoring
        ]
        
        # Create initial node
        initial_node = ReasoningNode(
            strategy=MagicMock(),
            prompt="Initial prompt",
            response="Initial response",
            cot=[{"action": "Initial", "content": "Let me think"}],
            parent=None
        )
        
        # Mock initialize_cot to return our initial node
        beam_search.initialize_cot = MagicMock(return_value=initial_node)
        
        # Create depth 1 strategies dict
        depth1_strategies_dict = {
            "Strategy1": {
                "score": 0.7,
                "strategy": strategy1,
                "cot": [{"action": "Step 1", "content": "First attempt"}],
                "prompt": "Prompt for Strategy1",
                "response": "Beam 1 depth 1"
            },
            "Strategy2": {
                "score": 0.6,
                "strategy": strategy2,
                "cot": [{"action": "Step 1", "content": "Alternative approach"}],
                "prompt": "Prompt for Strategy2",
                "response": "Beam 2 depth 1"
            }
        }
        
        # Create depth 2 strategies dict
        depth2_strategies_dict = {
            "Strategy1": {
                "score": 0.8,
                "strategy": strategy1,
                "cot": [{"action": "Final Conclusion", "content": "The answer is 4."}],
                "prompt": "Prompt for Strategy1",
                "response": "Beam 1 depth 2"
            },
            "Strategy2": {
                "score": 0.5,
                "strategy": strategy2,
                "cot": [{"action": "Final Conclusion", "content": "The answer is 5."}],
                "prompt": "Prompt for Strategy2",
                "response": "Beam 2 depth 2"
            }
        }
        
        # Mock the strategy_selector
        mock_selector = MagicMock()
        mock_strategy_selector_class.return_value = mock_selector
        
        # Configure the selector to return different strategies based on depth
        def select_side_effect(*args, **kwargs):
            depth = next((arg for arg in args if isinstance(arg, int)), 
                        kwargs.get('depth', 1))
            
            if depth == 1:
                return (
                    [strategy1, strategy2],
                    {
                        "strategies_dict": depth1_strategies_dict,
                        "scores": {"Strategy1": 0.7, "Strategy2": 0.6},
                    }
                )
            elif depth == 2:
                return (
                    [strategy1, strategy2],
                    {
                        "strategies_dict": depth2_strategies_dict,
                        "scores": {"Strategy1": 0.8, "Strategy2": 0.5},
                    }
                )
            else:
                return ([], {})
        
        mock_selector.select.side_effect = select_side_effect
        
        # Replace the strategy_selector on beam_search
        beam_search.strategy_selector = mock_selector
        
        # Create beams at depth 1
        beam1_depth1 = ReasoningNode(
            strategy=strategy1,
            prompt="Prompt for Strategy1",
            response="Beam 1 depth 1",
            cot=[{"action": "Step 1", "content": "First attempt"}],
            parent=initial_node
        )
        beam1_depth1.success = False
        beam1_depth1.is_final = False
        
        beam2_depth1 = ReasoningNode(
            strategy=strategy2,
            prompt="Prompt for Strategy2",
            response="Beam 2 depth 1",
            cot=[{"action": "Step 1", "content": "Alternative approach"}],
            parent=initial_node
        )
        beam2_depth1.success = False
        beam2_depth1.is_final = False
        
        # Mock initialize_beams to return our depth 1 beams
        beam_search.initialize_beams = MagicMock(return_value=[beam1_depth1, beam2_depth1])
        
        # Create successful node for depth 2
        beam1_depth2 = ReasoningNode(
            strategy=strategy1,
            prompt="Prompt for Strategy1",
            response="Beam 1 depth 2",
            cot=[{"action": "Final Conclusion", "content": "The answer is 4."}],
            parent=beam1_depth1
        )
        beam1_depth2.success = True
        beam1_depth2.is_final = True
        
        beam2_depth2 = ReasoningNode(
            strategy=strategy2,
            prompt="Prompt for Strategy2",
            response="Beam 2 depth 2",
            cot=[{"action": "Final Conclusion", "content": "The answer is 5."}],
            parent=beam2_depth1
        )
        beam2_depth2.success = False
        beam2_depth2.is_final = True
        
        # Mock verify_node for correct node verification
        def mock_verify_func(node, *args, **kwargs):
            if node.cot and len(node.cot) > 0 and "4" in str(node.cot):
                node.success = True
                node.is_final = True
                return True, "Correct!"
            return False, "Not correct"
        
        beam_search.verify_node = MagicMock(side_effect=mock_verify_func)
        
        # Run the search
        result = beam_search._search(
            question=question,
            ground_truth_answer=ground_truth_answer,
            search_llm=mock_llm_provider,
            scorer=mock_scorer,
            verifier=mock_verifier,
            strategy_registry=mock_registry
        )
        
        # Check result indicates success
        self.assertTrue(result.success)
        self.assertIsNotNone(result.terminal_nodes)
        self.assertTrue(any(node.success for node in result.terminal_nodes))
        
        # Verify that the search process was followed correctly
        beam_search.initialize_cot.assert_called_once()
        
        # Verify that at least one node was marked successful
        successful_nodes = [node for node in result.terminal_nodes if node.success]
        self.assertGreaterEqual(len(successful_nodes), 1)

    @patch('cot_forge.reasoning.strategies.ScoredStrategySelector')
    @patch('cot_forge.utils.parsing.extract_cot')
    @patch('cot_forge.reasoning.verifiers.BaseVerifier')
    @patch('cot_forge.llm.LLMProvider')
    def test_max_depth_reached(self,
                               mock_llm_provider,
                               mock_verifier,
                               mock_extract_cot,
                               mock_strategy_selector_class):
        """Test that search stops when max_depth is reached."""
        question = "What is 2 + 2?"
        ground_truth_answer = "4"
        max_depth = 4
        
        # Create SimpleBeamSearch instance with specific max_depth
        beam_search = SimpleBeamSearch(beam_width=2, branching_factor=2, max_depth=max_depth)
        
        # Set up mock responses for initial CoT + all depths
        mock_responses = ["Initial thinking"] + ["Beam response"] * (2 * max_depth)
        mock_llm_provider.generate.side_effect = mock_responses
        
        # Mock CoT extraction - all with incorrect answers
        cot_responses = [[{"action": "Initial", "content": "Let me think"}]]
        for i in range(2 * max_depth):
            cot_responses.append([{"action": "Step", "content": f"Thinking step {i}"}, 
                                {"action": "Final Conclusion", "content": "The answer is 5."}])
        mock_extract_cot.side_effect = cot_responses
        
        # Create strategy mocks
        strategy1 = MagicMock()
        strategy1.name = "Strategy1"
        strategy1.build_prompt = MagicMock(return_value="Prompt for Strategy1")
        strategy1.is_initial = False
        strategy1.minimum_depth = 1
        
        strategy2 = MagicMock()
        strategy2.name = "Strategy2"
        strategy2.build_prompt = MagicMock(return_value="Prompt for Strategy2")
        strategy2.is_initial = False
        strategy2.minimum_depth = 1
        
        # Mock strategy registry
        mock_registry = MagicMock()
        mock_registry.get_strategy.side_effect = lambda x: {
            "Strategy1": strategy1,
            "Strategy2": strategy2
        }[x]
        mock_registry.list_strategies.return_value = ["Strategy1", "Strategy2"]
        
        # Mock verifier to always fail
        mock_verifier.return_value = (False, "Incorrect")
        
        # Mock scorer
        mock_scorer = MagicMock(spec=BaseScorer)
        mock_scorer.return_value = {"Strategy1": 0.6, "Strategy2": 0.4}
        
        # Create initial node for beam search
        initial_node = ReasoningNode(
            strategy=MagicMock(),
            prompt="Initial prompt",
            response="Initial response",
            cot=cot_responses[0],
            parent=None
        )
        
        # Mock initialize_cot to return our predefined initial node
        beam_search.initialize_cot = MagicMock(return_value=initial_node)
        
        # Create mock strategies dict
        mock_strategies_dict = {
            "Strategy1": {
                "score": 0.6,
                "strategy": strategy1,
                "cot": [{"action": "Step", "content": "Thinking"}, 
                    {"action": "Final Conclusion", "content": "The answer is 5."}],
                "prompt": "Prompt for Strategy1",
                "response": "Strategy 1 response"
            },
            "Strategy2": {
                "score": 0.4,
                "strategy": strategy2,
                "cot": [{"action": "Step", "content": "Thinking"}, 
                    {"action": "Final Conclusion", "content": "The answer is 5."}],
                "prompt": "Prompt for Strategy2",
                "response": "Strategy 2 response"
            }
        }
        
        # Mock the strategy_selector
        mock_selector = MagicMock()
        mock_strategy_selector_class.return_value = mock_selector
        mock_selector.select.return_value = (
            [strategy1, strategy2],
            {
                "strategies_dict": mock_strategies_dict,
                "scores": {
                    "Strategy1": 0.6,
                    "Strategy2": 0.4,
                }
            }
        )
        
        # Replace the strategy_selector on beam_search
        beam_search.strategy_selector = mock_selector
        
        # Create beams that will expand until max depth
        beam1 = ReasoningNode(
            strategy=strategy1,
            prompt="Prompt for Strategy1",
            response="Beam 1 response",
            cot=[{"action": "Step", "content": "Thinking"}, 
                {"action": "Final Conclusion", "content": "The answer is 5."}],
            parent=initial_node
        )
        beam1.success = False
        beam1.is_final = False
        
        beam2 = ReasoningNode(
            strategy=strategy2,
            prompt="Prompt for Strategy2",
            response="Beam 2 response",
            cot=[{"action": "Step", "content": "Thinking"}, 
                {"action": "Final Conclusion", "content": "The answer is 5."}],
            parent=initial_node
        )
        beam2.success = False
        beam2.is_final = False
        
        # Mock initialize_beams to return our predefined beams
        beam_search.initialize_beams = MagicMock(return_value=[beam1, beam2])
        
        # Run the search
        result = beam_search._search(
            question=question,
            ground_truth_answer=ground_truth_answer,
            search_llm=mock_llm_provider,
            scorer=mock_scorer,
            verifier=mock_verifier,
            strategy_registry=mock_registry,
            max_depth=max_depth
        )
        
        # Verify the search was executed to max_depth
        self.assertEqual(mock_selector.select.call_count, max_depth)
        self.assertFalse(result.success)
        self.assertIsNotNone(result.terminal_nodes)


    @patch('cot_forge.reasoning.search.simple_beam_search.SimpleBeamSearch.initialize_cot')
    def test_initialization_error(self, mock_initialize_cot):
        """Test handling of initialization errors."""
        question = "What is 2 + 2?"
        ground_truth_answer = "4"
        
        # Mock initialization to raise an error
        mock_initialize_cot.side_effect = ValueError("Failed to initialize CoT")
        
        # Mock necessary objects
        mock_llm_provider = MagicMock()
        mock_scorer = MagicMock(spec=BaseScorer)
        mock_verifier = MagicMock()
        
        # Create the SimpleBeamSearch instance
        beam_search = SimpleBeamSearch()
        
        # Run the search algorithm
        result = beam_search._search(
            question=question,
            ground_truth_answer=ground_truth_answer,
            search_llm=mock_llm_provider,
            scorer=mock_scorer,
            verifier=mock_verifier,
        )
        
        # Check appropriate error handling
        self.assertFalse(result.success)
        self.assertTrue(len(result.terminal_nodes)==0)
        self.assertEqual(result.metadata["reason"], "Failed to initialize CoT")
        
    @patch('cot_forge.reasoning.verifiers.BaseVerifier')
    @patch('cot_forge.llm.LLMProvider')
    def test_beam_initialization_error(self,
                                       mock_llm_provider,
                                       mock_verifier):
        """Test handling of beam initialization errors."""
        question = "What is 2 + 2?"
        ground_truth_answer = "4"
        
        # Create an instance of SimpleBeamSearch
        beam_search = SimpleBeamSearch(beam_width=2, branching_factor=3)
        
        # Create initial node
        initial_node = ReasoningNode(
            strategy=MagicMock(),
            prompt="Initial prompt",
            response="Initial response",
            cot=[{"action": "Initial", "content": "Let me think"}],
            parent=None
        )
        
        # Mock necessary objects
        mock_registry = MagicMock()
        mock_scorer = MagicMock(spec=BaseScorer)
        
        # Mock evaluate_strategies to raise an exception
        beam_search.evaluate_strategies = MagicMock(side_effect=ValueError("Failed to evaluate strategies"))
        
        # Test the error handling
        with self.assertRaises(ValueError) as context:
            beam_search.initialize_beams(
                initial_node=initial_node,
                strategy_registry=mock_registry,
                scorer=mock_scorer,
                depth=1,
                search_llm=mock_llm_provider,
                question=question,
                ground_truth_answer=ground_truth_answer,
                verifier=mock_verifier,
                llm_kwargs={}
            )
        
        # Verify the error message
        self.assertEqual(str(context.exception), "Failed to select strategies")
        
    @patch('cot_forge.reasoning.strategies.ScoredStrategySelector')
    @patch('cot_forge.utils.parsing.extract_cot')
    @patch('cot_forge.reasoning.verifiers.BaseVerifier')
    @patch('cot_forge.llm.LLMProvider')
    def test_scorer_behavior(self,
                             mock_llm_provider,
                             mock_verifier,
                             mock_extract_cot,
                             mock_strategy_selector_class):
        """Test that beam search correctly uses scorer to select paths."""
        question = "What is 2 + 2?"
        ground_truth_answer = "4"
        
        # Set up mock responses
        mock_llm_provider.generate.side_effect = [
            "Initial thinking",
            "Strategy 1 response",
            "Strategy 2 response",
            "Strategy 3 response",
        ]
        
        # Mock CoT extraction
        mock_extract_cot.side_effect = [
            [{"action": "Initial", "content": "Let me think"}],
            [{"action": "Step 1", "content": "First approach"}],
            [{"action": "Step 1", "content": "Second approach"}],
            [{"action": "Step 1", "content": "Third approach"}],
        ]
        
        # Create strategy mocks with proper mock methods
        strategy1 = MagicMock()
        strategy1.name = "Strategy1"
        strategy1.build_prompt = MagicMock(return_value="Prompt for Strategy1")
        strategy1.is_initial = False
        strategy1.minimum_depth = 1
        
        strategy2 = MagicMock()
        strategy2.name = "Strategy2"
        strategy2.build_prompt = MagicMock(return_value="Prompt for Strategy2")
        strategy2.is_initial = False
        strategy2.minimum_depth = 1
        
        strategy3 = MagicMock()
        strategy3.name = "Strategy3"
        strategy3.build_prompt = MagicMock(return_value="Prompt for Strategy3")
        strategy3.is_initial = False
        strategy3.minimum_depth = 1
        
        # Mock strategy registry
        mock_registry = MagicMock()
        mock_registry.get_strategy.side_effect = lambda x: {
            "Strategy1": strategy1,
            "Strategy2": strategy2,
            "Strategy3": strategy3
        }[x]
        mock_registry.list_strategies.return_value = ["Strategy1", "Strategy2", "Strategy3"]
        
        # Mock verifier
        mock_verifier.return_value = (False, "Not yet correct")
        
        # Mock scorer to give clear preferences
        mock_scorer = MagicMock(spec=BaseScorer)
        mock_scorer.return_value = {
            "Strategy1": 0.9,  # Highest score
            "Strategy2": 0.4,
            "Strategy3": 0.7,  # Second highest
        }
        
        # Create an instance of SimpleBeamSearch with mocked methods
        beam_search = SimpleBeamSearch(beam_width=2, branching_factor=3)
        
        # Create initial node for testing
        initial_node = ReasoningNode(
            strategy=MagicMock(),
            prompt="Initial prompt",
            response="Initial response",
            cot=[{"action": "Initial", "content": "Let me think"}],
            parent=None
        )
        
        # Mock initialize_cot to return our predefined initial node
        beam_search.initialize_cot = MagicMock(return_value=initial_node)
        
        # Create mock strategies dict
        mock_strategies_dict = {
            "Strategy1": {
                "score": 0.9,
                "strategy": strategy1,
                "cot": [{"action": "Step 1", "content": "First approach"}],
                "prompt": "Prompt for Strategy1",
                "response": "Strategy 1 response"
            },
            "Strategy2": {
                "score": 0.4,
                "strategy": strategy2,
                "cot": [{"action": "Step 1", "content": "Second approach"}],
                "prompt": "Prompt for Strategy2",
                "response": "Strategy 2 response"
            },
            "Strategy3": {
                "score": 0.7,
                "strategy": strategy3,
                "cot": [{"action": "Step 1", "content": "Third approach"}],
                "prompt": "Prompt for Strategy3",
                "response": "Strategy 3 response"
            }
        }
        
        # Mock the strategy_selector
        mock_selector = MagicMock()
        mock_strategy_selector_class.return_value = mock_selector
        mock_selector.select.return_value = (
            [strategy1, strategy3],  # Top 2 strategies by score
            {
                "strategies_dict": mock_strategies_dict,
                "scores": {
                    "Strategy1": 0.9,
                    "Strategy2": 0.4,
                    "Strategy3": 0.7,
                }
            }
        )
        
        # Replace the strategy_selector on beam_search
        beam_search.strategy_selector = mock_selector
        
        # Create beam nodes that would be selected
        beam1 = ReasoningNode(
            strategy=strategy1,
            prompt="Prompt for Strategy1",
            response="Strategy 1 response",
            cot=[{"action": "Step 1", "content": "First approach"}],
            parent=initial_node
        )
        beam1.success = False
        beam1.is_final = True
        
        beam2 = ReasoningNode(
            strategy=strategy3,
            prompt="Prompt for Strategy3",
            response="Strategy 3 response",
            cot=[{"action": "Step 1", "content": "Third approach"}],
            parent=initial_node
        )
        beam2.success = False
        beam2.is_final = True
        
        # Mock the initialize_beams method to return our predefined beams
        beam_search.initialize_beams = MagicMock(return_value=[beam1, beam2])
        
        # Mock expand_beams to avoid the need for full simulation
        beam_search.expand_beams = MagicMock(return_value=[beam1, beam2])
        
        # Run the search
        result = beam_search._search(
            question=question,
            ground_truth_answer=ground_truth_answer,
            search_llm=mock_llm_provider,
            scorer=mock_scorer,
            verifier=mock_verifier,
            strategy_registry=mock_registry
        )
        
        # Verify the beam initialization was called
        beam_search.initialize_beams.assert_called_once()
        
        # Check the beams were created with the highest scoring strategies
        self.assertEqual(len(result.terminal_nodes), 2)
        
        # Get the strategies used in the beams
        beam_strategies = [node.strategy for node in result.terminal_nodes]
        
        # Verify the highest scoring strategies were selected
        self.assertEqual(set(beam_strategies), {strategy1, strategy3})
        self.assertTrue(strategy1 in beam_strategies)
        self.assertTrue(strategy3 in beam_strategies)
        self.assertTrue(strategy2 not in beam_strategies)