import unittest
from unittest.mock import MagicMock, patch

from cot_forge.llm import LLMProvider
from cot_forge.reasoning import NaiveLinearSearch, SimpleBeamSearch
from cot_forge.reasoning.scorers import BaseScorer
from cot_forge.reasoning.types import ReasoningNode
from cot_forge.reasoning.verifiers import LLMJudgeVerifier, BaseVerifier
from cot_forge.utils.parsing import extract_cot


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
            reasoning_llm=self.llm_provider,
            verifier=self.verifier
        )
                        
        # Verify LLM was called with correct parameters
        self.llm_provider.generate.assert_called_once()
        # Check the prompt contains the question
        self.assertIn(question, self.llm_provider.generate.call_args[1]['prompt'])
        
        # Verify the verifier was called with correct parameters
        self.verifier.assert_called_once()
        # Check that the node was passed to verifier
        verify_args = self.verifier.call_args
        self.assertEqual(verify_args[1]['question'], question)
        self.assertEqual(verify_args[1]['ground_truth_answer'], ground_truth_answer)
        
        # Check result structure and contents
        self.assertTrue(result['success'])
        self.assertIsNotNone(result['final_node'])
        self.assertEqual(result['final_answer'], "The answer is 4.")
        self.assertEqual(len(result['all_terminal_nodes']), 1)
        self.assertIn('depth', result['metadata'])
        
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
            reasoning_llm=self.llm_provider,
            verifier=self.verifier
        )
        
        # Check result indicates failure
        self.assertFalse(result['success'])
        self.assertIsNotNone(result['final_node'])
        self.assertEqual(result['final_answer'], "The answer is 5.")
        self.assertEqual(len(result['all_terminal_nodes']), 1)
        
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
            reasoning_llm=mock_llm_provider,
            verifier=mock_verifier,
            max_depth=2
        )
        
        # Check LLM was called twice
        self.assertEqual(mock_llm_provider.generate.call_count, 2)
        self.assertEqual(mock_verifier.call_count, 2)
        
        # Check result indicates success
        self.assertTrue(result['success'])
        self.assertIsNotNone(result['final_node'])
        self.assertEqual(result['final_answer'], "The answer is 4.")
        self.assertEqual(len(result['all_terminal_nodes']), 1)
        self.assertEqual(result['metadata']['depth'], 2)
        
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
            reasoning_llm = mock_llm_provider,
            verifier = mock_verifier,
        )
        
        # Check LLM was called exactly 3 times
        self.assertEqual(mock_llm_provider.generate.call_count, 3)
        self.assertEqual(mock_verifier.call_count, 3)
        
        # Check result indicates failure
        self.assertFalse(result['success'])
        self.assertIsNotNone(result['final_node'])
        self.assertEqual(len(result['all_terminal_nodes']), 1)
        self.assertEqual(result['metadata']['depth'], 3)
        
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
            llm_provider=mock_llm_provider,
            verifier=mock_verifier,
            reasoning_llm=mock_llm_provider,
        )
        
        # Check result indicates failure
        self.assertFalse(result['success'])
        self.assertEqual(len(result['all_terminal_nodes']), 1)
        self.assertIsNone(result['final_node'])
        self.assertIsNone(result['all_terminal_nodes'][0])
        self.assertIsNone(result['final_answer'])
        
    
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
            llm_provider=mock_llm_provider,
            verifier=mock_verifier,
            reasoning_llm=mock_llm_provider
        )
                                
        # Check result indicates failure
        self.assertFalse(result['success'])
        self.assertEqual(len(result['all_terminal_nodes']), 1)
        self.assertIsNone(result['final_node'])
        self.assertIsNone(result['all_terminal_nodes'][0])
        self.assertIsNone(result['final_answer'])
        self.assertIn('error', result['metadata']['reason'])
        
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
            reasoning_llm=self.llm_provider,
            verifier=mock_verifier,
            max_depth=max_depth
        )
    
        # Check result indicates issue with parsing
        self.assertFalse(result['success'])
        all(
            node.metadata.get('warning') == 'missing_final_conclusion' 
            for node in result['final_node'].get_full_node_chain()
        )
        self.assertIsNone(result['final_answer'])
        # Chain should have gone to max depth
        assert result['metadata']['depth'] == max_depth
        
            
class TestSimpleBeamSearch(unittest.TestCase):
    """Test that the simple beam search algorithm works as expected."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.search = MagicMock(spec=SimpleBeamSearch)
        self.llm_provider = MagicMock(spec=LLMProvider)
        self.verifier = MagicMock(spec=LLMJudgeVerifier)
        self.scorer = MagicMock(spec=BaseScorer)
        
    @patch('cot_forge.utils.parsing.extract_cot')
    @patch('cot_forge.reasoning.verifiers.BaseVerifier')
    @patch('cot_forge.llm.LLMProvider')
    def test_verification_success(self, mock_llm_provider, mock_verifier, mock_extract_cot):
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
        strategy1.build_prompt = MagicMock(return_value="Prompt for Strategy1")
        strategy1.is_initial = False
        strategy1.minimum_depth = 1
        
        strategy2 = MagicMock()
        strategy2.build_prompt = MagicMock(return_value="Prompt for Strategy2")
        strategy2.is_initial = False
        strategy2.minimum_depth = 1
        
        strategy3 = MagicMock()
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
        
        # Mock evaluate_strategies to return our pre-defined strategies tracker
        mock_strategies_tracker = {
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
        beam_search.evaluate_strategies = MagicMock(return_value=mock_strategies_tracker)
        
        # Also mock get_strategy_options
        beam_search.get_strategy_options = MagicMock(return_value=["Strategy1", "Strategy2", "Strategy3"])
        
        # Run the search
        result = beam_search._search(
            question=question,
            ground_truth_answer=ground_truth_answer,
            reasoning_llm=mock_llm_provider,
            scorer=mock_scorer,
            verifier=mock_verifier,
            strategy_registry=mock_registry
        )
        
        print("YAYA", result)
        
        # Check result indicates success
        self.assertTrue(result['success'])
        self.assertIsNotNone(result['all_terminal_nodes'])
        self.assertTrue(any(node.success for node in result['all_terminal_nodes']))
        
        # Verify the highest scoring strategies were selected
        # We need to get the nodes at depth 1 which are children of initial_node
        nodes_at_depth_1 = []
        for terminal_node in result['all_terminal_nodes']:
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
        
    def test_verification_failure(self):
        """Test handling of verification failure across all beams."""
        question = "What is 2 + 2?"
        ground_truth_answer = "4"
        
        # Create SimpleBeamSearch instance
        beam_search = SimpleBeamSearch(beam_width=2, branching_factor=2, max_depth=2)
        
        # Set up mock responses
        mock_llm_provider = MagicMock(spec=LLMProvider)
        mock_llm_provider.generate.side_effect = [
            # Initial CoT response
            "Initial thinking about 2+2",
            # Beam strategy responses
            "Strategy 1 response",
            "Strategy 2 response",
        ]
        
        # Mock CoT extraction
        mock_extract_cot = MagicMock()
        mock_extract_cot.side_effect = [
            [{"action": "Initial", "content": "Let me think about 2+2"}],  # Initial CoT
            [{"action": "Final Conclusion", "content": "The answer is 5."}],  # Strategy 1
            [{"action": "Final Conclusion", "content": "The answer is 3."}],  # Strategy 2
        ]
        
        # Mock strategy registry
        mock_registry = MagicMock()
        mock_registry.get_strategy.side_effect = lambda x: MagicMock(
            build_prompt=lambda question, previous_cot=None: f"Prompt for {x}",
            is_initial=False,
            minimum_depth=1
        )
        mock_registry.list_strategies.return_value = ["Strategy1", "Strategy2"]
        
        # Mock verifier to fail for all strategies
        mock_verifier = MagicMock(spec=BaseVerifier)
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
        
        # Mock get_strategy_options
        beam_search.get_strategy_options = MagicMock(return_value=["Strategy1", "Strategy2"])
        
        # Create a mock strategy tracker with scores
        mock_strategies_tracker = {
            "Strategy1": {
                "score": 0.6,
                "strategy": mock_registry.get_strategy("Strategy1"),
                "cot": [{"action": "Final Conclusion", "content": "The answer is 5."}],
                "prompt": "Prompt for Strategy1",
                "response": "Strategy 1 response"
            },
            "Strategy2": {
                "score": 0.4,
                "strategy": mock_registry.get_strategy("Strategy2"),
                "cot": [{"action": "Final Conclusion", "content": "The answer is 3."}],
                "prompt": "Prompt for Strategy2",
                "response": "Strategy 2 response"
            }
        }
        
        # Mock evaluate_strategies
        beam_search.evaluate_strategies = MagicMock(return_value=mock_strategies_tracker)
        
        # Patch extract_cot for the test duration
        with patch('cot_forge.utils.parsing.extract_cot', mock_extract_cot):
            # Run the search
            result = beam_search(
                question=question,
                ground_truth_answer=ground_truth_answer,
                reasoning_llm=mock_llm_provider,
                scorer=mock_scorer,
                verifier=mock_verifier,
                strategy_registry=mock_registry
            )
        
        # Check result indicates failure
        self.assertFalse(result['success'])
        self.assertIsNotNone(result['all_terminal_nodes'])
        self.assertFalse(any(node.success for node in result['all_terminal_nodes'] if node))
        
    @patch('cot_forge.utils.parsing.extract_cot')
    @patch('cot_forge.reasoning.verifiers.BaseVerifier')
    @patch('cot_forge.llm.LLMProvider')
    def test_multiple_depth_success(self, mock_llm_provider, mock_verifier, mock_extract_cot):
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
        strategy1.build_prompt = MagicMock(return_value="Prompt for Strategy1")
        strategy1.is_initial = False
        strategy1.minimum_depth = 1
        
        strategy2 = MagicMock()
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
        
        # Mock get_strategy_options
        beam_search.get_strategy_options = MagicMock(return_value=["Strategy1", "Strategy2"])
        
        # Create strategy trackers for different depths
        depth1_strategies = {
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
        
        depth2_strategies = {
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
        
        # Instead of using side_effect with a list, create a function that handles different calls
        def evaluate_strategies_mock(*args, **kwargs):
            depth = kwargs.get('depth', None)
            # For initialize_beams
            if depth == 1:
                return depth1_strategies
            # For the first expansion at depth 2
            elif depth == 2:
                # Mark one beam as successful so we don't try to expand further
                # This simulates the verification success for beam 1 at depth 2
                return depth2_strategies
            else:
                return {}  # Empty dict for any unexpected calls
                
        # Mock evaluate_strategies with our custom function
        beam_search.evaluate_strategies = MagicMock(side_effect=evaluate_strategies_mock)
        
        # Mock initialize_beams to create our initial beams
        beam1_depth1 = ReasoningNode(
            strategy=strategy1,
            prompt="Prompt for Strategy1",
            response="Beam 1 depth 1",
            cot=[{"action": "Step 1", "content": "First attempt"}],
            parent=initial_node
        )
        
        beam2_depth1 = ReasoningNode(
            strategy=strategy2,
            prompt="Prompt for Strategy2",
            response="Beam 2 depth 1",
            cot=[{"action": "Step 1", "content": "Alternative approach"}],
            parent=initial_node
        )
        
        initial_node.add_child(beam1_depth1)
        initial_node.add_child(beam2_depth1)
        
        beam_search.initialize_beams = MagicMock(return_value=[beam1_depth1, beam2_depth1])
        
        # Create a successful node for depth 2
        beam1_depth2 = ReasoningNode(
            strategy=strategy1,
            prompt="Prompt for Strategy1",
            response="Beam 1 depth 2",
            cot=[{"action": "Final Conclusion", "content": "The answer is 4."}],
            parent=beam1_depth1
        )
        # Mark it as successful
        beam1_depth2.success = True
        beam1_depth2.is_final = True
        
        # Also mock verify_node to correctly mark nodes as successful
        original_verify_node = beam_search.verify_node
        def mock_verify_node(node, *args, **kwargs):
            if node.cot and node.cot[0].get("action") == "Final Conclusion" and "4" in node.cot[0].get("content", ""):
                node.success = True
                node.is_final = True
                return True, "Correct"
            return False, "Not yet correct"
        beam_search.verify_node = MagicMock(side_effect=mock_verify_node)
        
        # Run the search
        result = beam_search._search(
            question=question,
            ground_truth_answer=ground_truth_answer,
            reasoning_llm=mock_llm_provider,
            scorer=mock_scorer,
            verifier=mock_verifier,
            strategy_registry=mock_registry
        )
        
        # Check result indicates success - using the SearchResult object attributes
        self.assertTrue(result['success'])
        self.assertIsNotNone(result['all_terminal_nodes'])
        self.assertTrue(any(node.success for node in result['all_terminal_nodes']))
        
        # Verify that the search process was followed correctly
        beam_search.initialize_cot.assert_called_once()
        
        # Verify that at least one node was marked successful
        successful_nodes = [node for node in result['all_terminal_nodes'] if node.success]
        self.assertGreaterEqual(len(successful_nodes), 1)
        
        # Verify that the successful node has the expected CoT
        # Note: We may need to adjust this if the exact node structure changes
        self.assertTrue(any(
            node.cot == [{"action": "Final Conclusion", "content": "The answer is 4."}]
            for node in result['all_terminal_nodes']
        ))
            
    @patch('cot_forge.utils.parsing.extract_cot')
    @patch('cot_forge.reasoning.verifiers.BaseVerifier')
    @patch('cot_forge.llm.LLMProvider')
    def test_max_depth_reached(self, mock_llm_provider, mock_verifier, mock_extract_cot):
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
        strategy1.build_prompt = MagicMock(return_value="Prompt for Strategy1")
        strategy1.is_initial = False
        strategy1.minimum_depth = 1
        
        strategy2 = MagicMock()
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
        
        # Mock get_strategy_options to return our predefined strategies
        beam_search.get_strategy_options = MagicMock(return_value=["Strategy1", "Strategy2"])
        
        # Create a mock strategy tracker with scores
        mock_strategies_tracker = {
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
        
        # Mock evaluate_strategies to return our pre-defined strategies tracker
        beam_search.evaluate_strategies = MagicMock(return_value=mock_strategies_tracker)
        
        # Run the search
        result = beam_search._search(
            question=question,
            ground_truth_answer=ground_truth_answer,
            reasoning_llm=mock_llm_provider,
            scorer=mock_scorer,
            verifier=mock_verifier,
            strategy_registry=mock_registry
        )
        
        # Verify the search was executed to max_depth
        self.assertEqual(beam_search.evaluate_strategies.call_count, max_depth - 1)
        self.assertFalse(result['success'])
        self.assertIsNotNone(result['all_terminal_nodes'])

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
            reasoning_llm=mock_llm_provider,
            scorer=mock_scorer,
            verifier=mock_verifier,
        )
        
        # Check appropriate error handling
        self.assertFalse(result['success'])
        self.assertIsNone(result['all_terminal_nodes'])
        self.assertEqual(result['metadata']["error"], "Failed to initialize CoT")
        
    @patch('cot_forge.utils.parsing.extract_cot')
    @patch('cot_forge.reasoning.verifiers.BaseVerifier')
    @patch('cot_forge.llm.LLMProvider')
    def test_beam_initialization_error(self, mock_llm_provider, mock_verifier, mock_extract_cot):
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
                reasoning_llm=mock_llm_provider,
                question=question,
                ground_truth_answer=ground_truth_answer,
                verifier=mock_verifier,
                llm_kwargs={}
            )
        
        # Verify the error message
        self.assertEqual(str(context.exception), "Failed to score strategies")
        
    @patch('cot_forge.utils.parsing.extract_cot')
    @patch('cot_forge.reasoning.verifiers.BaseVerifier')
    @patch('cot_forge.llm.LLMProvider')
    def test_scorer_behavior(self, mock_llm_provider, mock_verifier, mock_extract_cot):
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
        strategy1.build_prompt = MagicMock(return_value="Prompt for Strategy1")
        strategy1.is_initial = False
        strategy1.minimum_depth = 1
        
        strategy2 = MagicMock()
        strategy2.build_prompt = MagicMock(return_value="Prompt for Strategy2")
        strategy2.is_initial = False
        strategy2.minimum_depth = 1
        
        strategy3 = MagicMock()
        strategy3.build_prompt = MagicMock(return_value="Prompt for Strategy3")
        strategy3.is_initial = False
        strategy3.minimum_depth = 1
        
        # Mock strategy registry
        mock_registry = MagicMock()
        # Map strategy names to strategy mocks
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
        
        # Create a mock strategy tracker with scores - adding the 'strategy' key
        mock_strategies_tracker = {
            "Strategy1": {
                "score": 0.9,
                "strategy": strategy1,  # Add the strategy object
                "cot": [{"action": "Step 1", "content": "First approach"}],
                "prompt": "Prompt for Strategy1",
                "response": "Strategy 1 response"
            },
            "Strategy2": {
                "score": 0.4,
                "strategy": strategy2,  # Add the strategy object
                "cot": [{"action": "Step 1", "content": "Second approach"}],
                "prompt": "Prompt for Strategy2",
                "response": "Strategy 2 response"
            },
            "Strategy3": {
                "score": 0.7,
                "strategy": strategy3,  # Add the strategy object
                "cot": [{"action": "Step 1", "content": "Third approach"}],
                "prompt": "Prompt for Strategy3",
                "response": "Strategy 3 response"
            }
        }
        
        # Mock evaluate_strategies to return our pre-defined strategies tracker
        beam_search.evaluate_strategies = MagicMock(return_value=mock_strategies_tracker)
        
        # Also mock get_strategy_options for good measure
        beam_search.get_strategy_options = MagicMock(return_value=["Strategy1", "Strategy2", "Strategy3"])
        
        # Run the beam initialization with beam_width=2
        initial_node = ReasoningNode(
            strategy=MagicMock(),
            prompt="Initial prompt",
            response="Initial response",
            cot=[{"action": "Initial", "content": "Let me think"}],
            parent=None
        )
        
        beams = beam_search.initialize_beams(
            initial_node=initial_node,
            strategy_registry=mock_registry,
            scorer=mock_scorer,
            depth=1,
            reasoning_llm=mock_llm_provider,
            question=question,
            ground_truth_answer=ground_truth_answer,
            verifier=mock_verifier,
            llm_kwargs={}
        )
        
        # Check that the beams were created with the highest scoring strategies
        self.assertEqual(len(beams), 2)
        
        # Get the strategies used in the beams
        beam_strategies = [beam.strategy for beam in beams]
        
        # Verify the highest scoring strategies were selected
        self.assertTrue(strategy1 in beam_strategies)
        self.assertTrue(strategy3 in beam_strategies)
        self.assertTrue(strategy2 not in beam_strategies)