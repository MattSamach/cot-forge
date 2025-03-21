import unittest
from unittest.mock import MagicMock, Mock

from cot_forge.llm import LLMProvider
from cot_forge.reasoning.cot_builder import CoTBuilder
from cot_forge.reasoning.strategies import StrategyRegistry
from cot_forge.reasoning.types import ReasoningNode, SearchResult
from cot_forge.reasoning.verifiers import LLMJudgeVerifier


class TestCoTBuilder(unittest.TestCase):
    """Tests for the CoTBuilder class."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_llm = Mock(spec=LLMProvider)
        self.mock_search = Mock()
        self.mock_strategy_reg = Mock(spec=StrategyRegistry)
        self.mock_verifier = Mock(spec=LLMJudgeVerifier)
        
        # Create a mock search result to be returned by the search algorithm
        self.mock_search_result: SearchResult = {
            'terminal_nodes': [MagicMock(spec=ReasoningNode)],
            'success': True,
            'final_answer': 'Mock answer',
            'metadata': {'test': 'metadata'}
        }
        
        # Configure the mock search to return the mock result
        self.mock_search.return_value = self.mock_search_result
        
        # Create the CoTBuilder with mocked dependencies
        self.cot_builder = CoTBuilder(
            reasoning_llm=self.mock_llm,
            search=self.mock_search,
            strategy_reg=self.mock_strategy_reg,
            verifier=self.mock_verifier
        )

    def test_build(self):
        """Test the build method."""
        # Define test inputs
        question = "What is 2 + 2?"
        ground_truth = "4"
        llm_kwargs = {'temperature': 0.7}
        extra_kwargs = {'max_depth': 3, 'custom_param': 'value'}
        
        # Call the build method
        result = self.cot_builder.build(
            question=question,
            ground_truth_answer=ground_truth,
            llm_kwargs=llm_kwargs,
            **extra_kwargs
        )
        
        # Assert the search method was called with correct parameters
        self.mock_search.assert_called_once_with(
            question=question,
            ground_truth_answer=ground_truth,
            reasoning_llm=self.mock_llm,
            llm_kwargs=llm_kwargs,
            strategy_registry=self.mock_strategy_reg,
            verifier=self.mock_verifier,
            scorer=None,
            max_depth=3,
            custom_param='value'
        )
        
        # Assert the result is as expected
        self.assertEqual(result, self.mock_search_result)

    def test_build_batch_single_thread(self):
        """Test the build_batch method in single-threaded mode."""
        # Define test inputs
        questions = ["What is 2 + 2?", "What is 3 + 3?"]
        ground_truths = ["4", "6"]
        llm_kwargs = {'temperature': 0.7}
        
        # Call the build_batch method
        results = self.cot_builder.build_batch(
            questions=questions,
            ground_truth_answers=ground_truths,
            llm_kwargs=llm_kwargs,
            multi_thread=False,
            progress_bar=False
        )
        
        # Assert the search method was called for each question
        self.assertEqual(self.mock_search.call_count, 2)
        
        # Check parameters passed for first call
        first_call_args = self.mock_search.call_args_list[0][1]
        self.assertEqual(first_call_args['question'], questions[0])
        self.assertEqual(first_call_args['ground_truth_answer'], ground_truths[0])
        self.assertEqual(first_call_args['reasoning_llm'], self.mock_llm)
        self.assertEqual(first_call_args['llm_kwargs'], llm_kwargs)
        
        # Check parameters passed for second call
        second_call_args = self.mock_search.call_args_list[1][1]
        self.assertEqual(second_call_args['question'], questions[1])
        self.assertEqual(second_call_args['ground_truth_answer'], ground_truths[1])
        
        # Assert the results are as expected
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0], self.mock_search_result)
        self.assertEqual(results[1], self.mock_search_result)

    def test_build_batch_multi_thread(self):
        """Test the build_batch method in multi-threaded mode."""
        # Define test inputs
        questions = ["What is 2 + 2?", "What is 3 + 3?", "What is 4 + 4?"]
        ground_truths = ["4", "6", "8"]
        llm_kwargs = {'temperature': 0.7}
        max_workers = 2
        
        # Call the build_batch method with multi_thread=True
        results = self.cot_builder.build_batch(
            questions=questions,
            ground_truth_answers=ground_truths,
            llm_kwargs=llm_kwargs,
            multi_thread=True,
            progress_bar=False,
            max_workers=max_workers
        )
        
        # Assert the search method was called for each question
        self.assertEqual(self.mock_search.call_count, 3)
        
        # Get the parameters passed for each call
        call_args_list = self.mock_search.call_args_list
        
        # Create a set of tuples (question, ground_truth) for each call
        call_params = {
            (call_args[1]['question'], call_args[1]['ground_truth_answer'])
            for call_args in call_args_list
        }
        
        # Assert all question-answer pairs were processed
        expected_params = set(zip(questions, ground_truths))
        self.assertEqual(call_params, expected_params)
        
        # Assert the results are as expected
        self.assertEqual(len(results), 3)
        for result in results:
            self.assertEqual(result, self.mock_search_result)

    def test_mismatched_questions_answers(self):
        """Test that an error is raised when questions and answers have different lengths."""
        questions = ["What is 2 + 2?", "What is 3 + 3?"]
        ground_truths = ["4"]  # Only one answer
        
        with self.assertRaises(ValueError):
            self.cot_builder.build_batch(
                questions=questions,
                ground_truth_answers=ground_truths
            )

    def test_multi_thread_missing_max_workers(self):
        """Test that an error is raised when multi_thread=True but max_workers is not specified."""
        questions = ["What is 2 + 2?"]
        ground_truths = ["4"]
        
        with self.assertRaises(ValueError):
            self.cot_builder.build_batch(
                questions=questions,
                ground_truth_answers=ground_truths,
                multi_thread=True,
                max_workers=None
            )

    def test_empty_batch(self):
        """Test build_batch with empty lists."""
        results = self.cot_builder.build_batch(
            questions=[],
            ground_truth_answers=[],
            progress_bar=False
        )
        
        # Search should not have been called
        self.mock_search.assert_not_called()
        
        # Results should be an empty list
        self.assertEqual(results, [])

    def test_repr_and_str(self):
        """Test the __repr__ and __str__ methods."""
        self.assertIn("CoTBuilder", repr(self.cot_builder))
        self.assertIn("CoTBuilder", str(self.cot_builder))
        self.assertIn(str(self.mock_llm), str(self.cot_builder))
        self.assertIn(str(self.mock_search), str(self.cot_builder))


if __name__ == '__main__':
    unittest.main()
