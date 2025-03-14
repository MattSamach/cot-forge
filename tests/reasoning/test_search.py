import unittest
from unittest.mock import MagicMock, patch

from cot_forge.reasoning import naive_linear_search
from cot_forge.reasoning.scorers import BaseScorer
from cot_forge.reasoning.search.simple_beam_search import (initialize_beams,
                                                           simple_beam_search)
from cot_forge.reasoning.types import ReasoningNode
from cot_forge.reasoning.verifiers import LLMJudgeVerifier


class TestNaiveLinearSearch(unittest.TestCase):
    """Test that the naive linear search algorithm works as expected."""
    
    @patch('cot_forge.reasoning.verifiers.LLMJudgeVerifier')
    @patch('cot_forge.llm.LLMProvider')
    def test_verification_success(self, mock_llm_provider, mock_verifier):
        """Test that the search algorithm returns a successful result."""
        question = "What is 2 + 2?"
        ground_truth_answer = "4"
        
        # Set up the mock response
        mock_llm_provider.generate.return_value = """{
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
        mock_verifier.return_value = True, "The answer is correct."

        # Run the search algorithm
        result = naive_linear_search(
            question=question,
            ground_truth_answer=ground_truth_answer,
            llm_provider=mock_llm_provider,
            verifier=mock_verifier
        )
                        
        # Verify LLM was called with correct parameters
        mock_llm_provider.generate.assert_called_once()
        # Check the prompt contains the question
        self.assertIn(question, mock_llm_provider.generate.call_args[1]['prompt'])
        
        # Verify the verifier was called with correct parameters
        mock_verifier.assert_called_once()
        # Check that the node was passed to verifier
        verify_args = mock_verifier.call_args
        self.assertEqual(verify_args[1]['question'], question)
        self.assertEqual(verify_args[1]['ground_truth_answer'], ground_truth_answer)
        
        # Check result structure and contents
        self.assertTrue(result['success'])
        self.assertIsNotNone(result['final_node'])
        self.assertEqual(result['final_answer'], "The answer is 4.")
        self.assertEqual(len(result['all_terminal_nodes']), 1)
        self.assertIn('depth', result['metadata'])
        
    @patch('cot_forge.reasoning.verifiers.LLMJudgeVerifier')
    @patch('cot_forge.llm.LLMProvider')
    def test_verification_failure(self, mock_llm_provider, mock_verifier):
        """Test handling of verification failure."""
        question = "What is 2 + 2?"
        ground_truth_answer = "4"
        
        # Set up the mock response with an incorrect answer
        mock_llm_provider.generate.return_value = """{
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
        mock_verifier.return_value = False, "The answer is incorrect."

        # Run the search algorithm with default max_depth=1
        result = naive_linear_search(
            question=question,
            ground_truth_answer=ground_truth_answer,
            llm_provider=mock_llm_provider,
            verifier=mock_verifier
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
        result = naive_linear_search(
            question=question,
            ground_truth_answer=ground_truth_answer,
            llm_provider=mock_llm_provider,
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
                },
                {
                    "action": "Verification",
                    "content": "The answer is correct because 2 + 2 = 5."
                }
            ]
        }"""
        mock_llm_provider.generate.return_value = incorrect_response
        mock_verifier.return_value = False, "The answer is incorrect."

        # Run the search algorithm with max_depth=3
        result = naive_linear_search(
            question=question,
            ground_truth_answer=ground_truth_answer,
            llm_provider=mock_llm_provider,
            verifier=mock_verifier,
            max_depth=3
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
        result = naive_linear_search(
            question=question,
            ground_truth_answer=ground_truth_answer,
            llm_provider=mock_llm_provider,
            verifier=mock_verifier
        )
        
        # Check result indicates failure
        self.assertFalse(result['success'])
        self.assertEqual(len(result['all_terminal_nodes']), 1)
        self.assertIsNone(result['final_node'])
        self.assertIsNone(result['all_terminal_nodes'][0])
        self.assertIsNone(result['final_answer'])
        self.assertIn('error', result['metadata'])
        
    
    @patch('cot_forge.reasoning.verifiers.LLMJudgeVerifier')
    @patch('cot_forge.llm.LLMProvider')
    def test_llm_exception(self, mock_llm_provider, mock_verifier):
        """Test handling of LLM exception."""
        question = "What is 2 + 2?"
        ground_truth_answer = "4"
        
        # Set up mock to raise an exception
        mock_llm_provider.generate.side_effect = Exception("API error")
        
        # Run the search algorithm
        result = naive_linear_search(
            question=question,
            ground_truth_answer=ground_truth_answer,
            llm_provider=mock_llm_provider,
            verifier=mock_verifier
        )
                                
        # Check result indicates failure
        self.assertFalse(result['success'])
        self.assertEqual(len(result['all_terminal_nodes']), 1)
        self.assertIsNone(result['final_node'])
        self.assertIsNone(result['all_terminal_nodes'][0])
        self.assertIsNone(result['final_answer'])
        self.assertIn('error', result['metadata'])
        self.assertIn('API error', result['metadata']['error'])
        
    @patch('cot_forge.reasoning.verifiers.LLMJudgeVerifier')
    @patch('cot_forge.llm.LLMProvider')
    def test_missing_final_conclusion(self, mock_llm_provider, mock_verifier):
        """Test handling of LLM response without Final Conclusion."""
        
        verifier = LLMJudgeVerifier()
        question = "What is 2 + 2?"
        ground_truth_answer = "4"
        
        # Set up the mock response without Final Conclusion
        mock_llm_provider.generate.return_value = """{
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
        # Run the search algorithm
        result = naive_linear_search(
            question=question,
            ground_truth_answer=ground_truth_answer,
            llm_provider=mock_llm_provider,
            verifier=verifier,
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
    
    @patch('cot_forge.reasoning.search.simple_beam_search.extract_cot')
    @patch('cot_forge.reasoning.search.simple_beam_search.get_strategy_options')
    @patch('cot_forge.reasoning.verifiers.BaseVerifier')
    @patch('cot_forge.llm.LLMProvider')
    def test_verification_success(self, mock_llm_provider, mock_verifier, mock_get_strategy, mock_extract_cot):
        """Test that the search algorithm returns a successful result when a beam succeeds."""
        question = "What is 2 + 2?"
        ground_truth_answer = "4"
        
        # Set up mock responses
        mock_llm_provider.generate.side_effect = [
            # Initial CoT response
            "Initial thinking about 2+2",
            # First beam strategy responses
            "Strategy 1 response",
            "Strategy 2 response",
            "Strategy 3 response"
        ]
        
        # Mock CoT extraction
        mock_extract_cot.side_effect = [
            [{"action": "Initial", "content": "Let me think about 2+2"}],  # Initial CoT
            [{"action": "Final Conclusion", "content": "The answer is 4."}],  # Strategy 1
            [{"action": "Final Conclusion", "content": "The answer is 5."}],  # Strategy 2
            [{"action": "Final Conclusion", "content": "The answer is 4."}],  # Strategy 3
        ]
        
        # Mock strategy options
        mock_get_strategy.return_value = ["Strategy1", "Strategy2", "Strategy3"]
        
        # Mock strategy registry
        mock_registry = MagicMock()
        mock_registry.get_strategy.side_effect = lambda x: MagicMock(
            build_prompt=lambda question, previous_cot=None: f"Prompt for {x}",
            is_initial=False,
            minimum_depth=1
        )
        mock_registry.list_strategies.return_value = ["Strategy1", "Strategy2", "Strategy3"]
        
        # Mock verifier to succeed for Strategy 3
        mock_verifier.side_effect = [
            (False, "Not correct yet"),  # Strategy 1
            (False, "Incorrect"),        # Strategy 2
            (True, "Correct!"),          # Strategy 3
        ]
        
        # Mock scorer
        mock_scorer = MagicMock(spec=BaseScorer)
        mock_scorer.return_value = {
            "Strategy1": 0.7,
            "Strategy2": 0.5,
            "Strategy3": 0.9
        }
        
        # Run the search algorithm
        result = simple_beam_search(
            question=question,
            ground_truth_answer=ground_truth_answer,
            llm_provider=mock_llm_provider,
            scorer=mock_scorer,
            verifier=mock_verifier,
            strategy_registry=mock_registry,
            beam_width=3,
            branching_factor=3,
            max_depth=2
        )
        
        # Check result indicates success
        self.assertTrue(result['success'])
        self.assertIsNotNone(result['all_terminal_nodes'])
        self.assertTrue(any(node.success for node in result['all_terminal_nodes']))
        
    @patch('cot_forge.reasoning.search.simple_beam_search.extract_cot')
    @patch('cot_forge.reasoning.search.simple_beam_search.get_strategy_options')
    @patch('cot_forge.reasoning.verifiers.BaseVerifier')
    @patch('cot_forge.llm.LLMProvider')
    def test_verification_failure(self, mock_llm_provider, mock_verifier, mock_get_strategy, mock_extract_cot):
        """Test handling of verification failure across all beams."""
        question = "What is 2 + 2?"
        ground_truth_answer = "4"
        
        # Set up mock responses
        mock_llm_provider.generate.side_effect = [
            # Initial CoT response
            "Initial thinking about 2+2",
            # Beam strategy responses
            "Strategy 1 response",
            "Strategy 2 response",
        ]
        
        # Mock CoT extraction
        mock_extract_cot.side_effect = [
            [{"action": "Initial", "content": "Let me think about 2+2"}],  # Initial CoT
            [{"action": "Final Conclusion", "content": "The answer is 5."}],  # Strategy 1
            [{"action": "Final Conclusion", "content": "The answer is 3."}],  # Strategy 2
        ]
        
        # Mock strategy options
        mock_get_strategy.return_value = ["Strategy1", "Strategy2"]
        
        # Mock strategy registry
        mock_registry = MagicMock()
        mock_registry.get_strategy.side_effect = lambda x: MagicMock(
            build_prompt=lambda question, previous_cot=None: f"Prompt for {x}",
            is_initial=False,
            minimum_depth=1
        )
        mock_registry.list_strategies.return_value = ["Strategy1", "Strategy2"]
        
        # Mock verifier to fail for all strategies
        mock_verifier.return_value = (False, "Incorrect answer")
        
        # Mock scorer
        mock_scorer = MagicMock(spec=BaseScorer)
        mock_scorer.return_value = {
            "Strategy1": 0.6,
            "Strategy2": 0.4,
        }
        
        # Run the search algorithm
        result = simple_beam_search(
            question=question,
            ground_truth_answer=ground_truth_answer,
            llm_provider=mock_llm_provider,
            scorer=mock_scorer,
            verifier=mock_verifier,
            strategy_registry=mock_registry,
            beam_width=2,
            branching_factor=2,
            max_depth=2
        )
        
        # Check result indicates failure
        self.assertFalse(result['success'])
        self.assertIsNotNone(result['all_terminal_nodes'])
        self.assertFalse(any(node.success for node in result['all_terminal_nodes']))
        
    @patch('cot_forge.reasoning.search.simple_beam_search.extract_cot')
    @patch('cot_forge.reasoning.search.simple_beam_search.generate_cot')
    @patch('cot_forge.reasoning.search.simple_beam_search.get_strategy_options')
    @patch('cot_forge.reasoning.verifiers.BaseVerifier')
    @patch('cot_forge.llm.LLMProvider')
    def test_multiple_depth_success(self, mock_llm_provider, mock_verifier, mock_get_strategy, mock_generate_cot, mock_extract_cot):
        """Test successful search after multiple depths."""
        question = "What is 2 + 2?"
        ground_truth_answer = "4"
        
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
        
        # Mock strategy options
        mock_get_strategy.return_value = ["Strategy1", "Strategy2"]
        
        # Mock strategy registry
        mock_registry = MagicMock()
        mock_registry.get_strategy.side_effect = lambda x: MagicMock(
            build_prompt=lambda question, previous_cot=None: f"Prompt for {x}",
            is_initial=False,
            minimum_depth=1
        )
        mock_registry.list_strategies.return_value = ["Strategy1", "Strategy2"]
        
        # Mock verifier to succeed only at depth 2 for first beam
        mock_verifier.side_effect = [
            (False, "Not yet"),  # Beam 1 depth 1
            (False, "Not yet"),  # Beam 2 depth 1
            (True, "Correct!"),  # Beam 1 depth 2
            (False, "Wrong"),    # Beam 2 depth 2
        ]
        
        # Mock scorer with consistent scoring
        mock_scorer = MagicMock(spec=BaseScorer)
        mock_scorer.side_effect = [
            {"Strategy1": 0.7, "Strategy2": 0.6},  # Depth 1
            {"Strategy1": 0.8, "Strategy2": 0.5},  # Depth 2
        ]
        
        # Use a return_value instead of side_effect to handle any number of calls
        # This ensures generate_cot always returns a valid response regardless of how many times it's called
        mock_generate_cot.return_value = ("response", [{"action": "Final Conclusion", "content": "The answer is 4."}], "prompt")
        
        # Run the search algorithm
        result = simple_beam_search(
            question=question,
            ground_truth_answer=ground_truth_answer,
            llm_provider=mock_llm_provider,
            scorer=mock_scorer,
            verifier=mock_verifier,
            strategy_registry=mock_registry,
            beam_width=2,
            branching_factor=2,
            max_depth=3
        )
        
        # Check result indicates success
        self.assertTrue(result['success'])
        self.assertEqual(len(result['all_terminal_nodes']), 2)
        self.assertTrue(any(node.success for node in result['all_terminal_nodes']))
            
    @patch('cot_forge.reasoning.search.simple_beam_search.extract_cot')
    @patch('cot_forge.reasoning.search.simple_beam_search.generate_cot')
    @patch('cot_forge.reasoning.search.simple_beam_search.get_strategy_options')
    @patch('cot_forge.reasoning.verifiers.BaseVerifier')
    @patch('cot_forge.llm.LLMProvider')
    def test_max_depth_reached(self, mock_llm_provider, mock_verifier, mock_get_strategy, mock_generate_cot, mock_extract_cot):
        """Test that search stops when max_depth is reached."""
        question = "What is 2 + 2?"
        ground_truth_answer = "4"
        max_depth = 4
        
        # Set up mock responses for initial CoT + all depths
        mock_responses = ["Initial thinking"] + ["Beam response"] * (2 * max_depth)
        mock_llm_provider.generate.side_effect = mock_responses
        
        # Mock CoT extraction - all with incorrect answers
        cot_responses = [[{"action": "Initial", "content": "Let me think"}]]
        for i in range(2 * max_depth):
            cot_responses.append([{"action": "Step", "content": f"Thinking step {i}"}, 
                                {"action": "Final Conclusion", "content": "The answer is 5."}])
        mock_extract_cot.side_effect = cot_responses
        
        # Mock strategy options
        mock_get_strategy.return_value = ["Strategy1", "Strategy2"]
        
        # Mock strategy registry
        mock_registry = MagicMock()
        mock_registry.get_strategy.side_effect = lambda x: MagicMock(
            build_prompt=lambda question, previous_cot=None: f"Prompt for {x}",
            is_initial=False,
            minimum_depth=1
        )
        mock_registry.list_strategies.return_value = ["Strategy1", "Strategy2"]
        
        # Mock verifier to always fail
        mock_verifier.return_value = (False, "Incorrect")
        
        # Mock scorer
        mock_scorer = MagicMock(spec=BaseScorer)
        mock_scorer.return_value = {"Strategy1": 0.6, "Strategy2": 0.4}
        
        # Mock generate_cot to return valid responses
        mock_generate_cot.return_value = ("response", [{"action": "Final Conclusion", "content": "The answer is 5."}], "prompt")
        
        # Run the search algorithm
        result = simple_beam_search(
            question=question,
            ground_truth_answer=ground_truth_answer,
            llm_provider=mock_llm_provider,
            scorer=mock_scorer,
            verifier=mock_verifier,
            strategy_registry=mock_registry,
            beam_width=2,
            branching_factor=2,
            max_depth=max_depth
        )
        
        # Verify that max depth was reached with correct number of verifier calls
        # Should be called twice at each depth (depth 2, depth 3, depth 4) = 6 times
        self.assertEqual(mock_verifier.call_count, 2 * (max_depth-1))
        self.assertFalse(result['success'])

    @patch('cot_forge.reasoning.search.simple_beam_search.initialize_cot')
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
        
        # Run the search algorithm
        result = simple_beam_search(
            question=question,
            ground_truth_answer=ground_truth_answer,
            llm_provider=mock_llm_provider,
            scorer=mock_scorer,
            verifier=mock_verifier,
        )
        
        # Check appropriate error handling
        self.assertFalse(result['success'])
        self.assertIsNone(result['all_terminal_nodes'])
        self.assertEqual(result['metadata']["error"], "Failed to initialize CoT")
        
    @patch('cot_forge.reasoning.search.simple_beam_search.extract_cot')
    @patch('cot_forge.reasoning.search.simple_beam_search.get_strategy_options')
    @patch('cot_forge.reasoning.search.simple_beam_search.initialize_cot')
    def test_beam_initialization_error(self, mock_initialize_cot, mock_get_strategy, mock_extract_cot):
        """Test handling of beam initialization errors."""
        question = "What is 2 + 2?"
        ground_truth_answer = "4"
        
        # Mock CoT initialization to succeed but beam initialization to fail
        mock_initialize_cot.return_value = ReasoningNode(
            strategy=MagicMock(),
            prompt="Initial prompt",
            response="Initial response",
            cot=[{"action": "Initial", "content": "Let me think"}],
            parent=None
        )
        
        # Mock get_strategy_options to fail
        mock_get_strategy.side_effect = ValueError("Failed to get strategy options")
        
        # Mock necessary objects
        mock_llm_provider = MagicMock()
        mock_scorer = MagicMock(spec=BaseScorer)
        mock_verifier = MagicMock()
        mock_registry = MagicMock()
        
        # Run the search algorithm
        result = simple_beam_search(
            question=question,
            ground_truth_answer=ground_truth_answer,
            llm_provider=mock_llm_provider,
            scorer=mock_scorer,
            verifier=mock_verifier,
            strategy_registry=mock_registry
        )
        
        # Check appropriate error handling
        self.assertFalse(result['success'])
        self.assertIsNone(result['all_terminal_nodes'])
        self.assertEqual(result['metadata']["error"], "Failed to initialize beams")
        
    @patch('cot_forge.reasoning.search.simple_beam_search.extract_cot')
    @patch('cot_forge.reasoning.search.simple_beam_search.get_strategy_options')
    @patch('cot_forge.reasoning.verifiers.BaseVerifier')
    @patch('cot_forge.llm.LLMProvider')
    def test_scorer_behavior(self, mock_llm_provider, mock_verifier, mock_get_strategy, mock_extract_cot):
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
        
        # Mock strategy options
        mock_get_strategy.return_value = ["Strategy1", "Strategy2", "Strategy3"]
        
        # Mock strategy registry
        mock_registry = MagicMock()
        
        # Create strategy mocks with proper mock methods instead of lambdas
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
        
        # Run the beam initialization with beam_width=2
        initial_node = ReasoningNode(
            strategy=MagicMock(),
            prompt="Initial prompt",
            response="Initial response",
            cot=[{"action": "Initial", "content": "Let me think"}],
            parent=None
        )
        
        beams = initialize_beams(
            initial_node=initial_node,
            strategy_registry=mock_registry,
            beam_width=2,
            scorer=mock_scorer,
            branching_factor=3,
            depth=1,
            llm_provider=mock_llm_provider,
            question=question,
            ground_truth_answer=ground_truth_answer,
            verifier=mock_verifier
        )
        
        # Check that the beams were created with the highest scoring strategies
        self.assertEqual(len(beams), 2)
        
        # Get the strategies used in the beams
        beam_strategies = [beam.strategy for beam in beams]
        
        # Verify the highest scoring strategies were selected
        self.assertTrue(strategy1 in beam_strategies)
        self.assertTrue(strategy3 in beam_strategies)
        self.assertTrue(strategy2 not in beam_strategies)