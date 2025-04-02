import json
import tempfile
import unittest
from unittest.mock import MagicMock, Mock, patch

from cot_forge.llm import LLMProvider
from cot_forge.persistence import PersistenceManager
from cot_forge.reasoning.cot_builder import CoTBuilder
from cot_forge.reasoning.search.search_algorithm import SearchAlgorithm
from cot_forge.reasoning.strategies import StrategyRegistry
from cot_forge.reasoning.types import ReasoningNode, SearchResult
from cot_forge.reasoning.verifiers import BaseVerifier, LLMJudgeVerifier


class TestCoTBuilder(unittest.TestCase):
    """Tests for the CoTBuilder class."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_llm = Mock(spec=LLMProvider)
        self.mock_search = Mock()
        self.mock_strategy_registry = Mock(spec=StrategyRegistry)
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
            search_llm=self.mock_llm,
            search=self.mock_search,
            strategy_registry=self.mock_strategy_registry,
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
            search_llm=self.mock_llm,
            llm_kwargs=llm_kwargs,
            strategy_registry=self.mock_strategy_registry,
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
        self.assertEqual(first_call_args['search_llm'], self.mock_llm)
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
        expected_params = set(zip(questions, ground_truths, strict=False))
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

class TestCoTBuilderPersistence(unittest.TestCase):
    """Tests for the persistence functionality of CoTBuilder."""

    def setUp(self):
        """Set up test fixtures."""
        # Mock the core components
        self.mock_llm = Mock(spec=LLMProvider)
        self.mock_llm.to_dict = Mock(return_value={"type": "mock_llm"})
        
        self.mock_search = Mock(spec=SearchAlgorithm)
        self.mock_search.name = "MockSearch"
        self.mock_search.to_dict = Mock(return_value={"type": "mock_search"})
        
        self.mock_verifier = Mock(spec=BaseVerifier)
        self.mock_verifier.to_dict = Mock(return_value={"type": "mock_verifier"})
        
        self.mock_strategy_registry = Mock(spec=StrategyRegistry)
        self.mock_strategy_registry.serialize = Mock(return_value={"strategies": ["mock"]})
        self.mock_strategy_registry.list_strategies = Mock(return_value=["mock_strategy"])
        
        # Create a mock search result
        self.mock_search_result = Mock(spec=SearchResult)
        self.mock_search_result.success = True
        self.mock_search_result.serialize = Mock(return_value={"success": True, "mock": "data"})
        
        # Configure the mock search to return the mock result
        self.mock_search.return_value = self.mock_search_result
        
        # Create a temporary directory for persistence
        self.temp_dir = tempfile.TemporaryDirectory()
        
        # Create a real PersistenceManager with the temp directory
        self.dataset_name = "test_dataset"
        self.persistence = PersistenceManager(
            dataset_name=self.dataset_name,
            search_name=self.mock_search.name,
            base_dir=self.temp_dir.name,
            auto_resume=False
        )

    def tearDown(self):
        """Clean up after tests."""
        self.temp_dir.cleanup()

    def test_with_persistence_factory_method(self):
        """Test the factory method that creates a CoTBuilder with persistence."""
        with patch("cot_forge.reasoning.cot_builder.PersistenceManager") as mock_pm_class:
            mock_pm = MagicMock()
            mock_pm_class.return_value = mock_pm
            
            # Create a builder with persistence
            builder = CoTBuilder.with_persistence(
                search_llm=self.mock_llm,
                search=self.mock_search,
                verifier=self.mock_verifier,
                dataset_name=self.dataset_name,
                base_dir="test_dir",
                auto_resume=True
            )
            
            # Assert PersistenceManager was created with correct parameters
            mock_pm_class.assert_called_once_with(
                dataset_name=self.dataset_name,
                search_name=self.mock_search.name,
                base_dir="test_dir",
                auto_resume=True
            )
            
            # Assert the builder has the persistence object
            self.assertEqual(builder.persistence, mock_pm)
            
            # Assert save_config was called
            mock_pm.save_config.assert_called_once_with(builder)

    def test_save_config(self):
        """Test that configuration is saved when CoTBuilder is created with persistence."""
        # Mock the save_config method
        self.persistence.save_config = MagicMock()
        
        # Create the builder with persistence
        builder = CoTBuilder(
            search_llm=self.mock_llm,
            search=self.mock_search,
            verifier=self.mock_verifier,
            persistence=self.persistence
        )
        
        # Assert save_config was called with the builder instance
        self.persistence.save_config.assert_called_once_with(builder)

    def test_skip_processed_questions(self):
        """Test that already processed questions are skipped."""
        # Mock persistence methods
        self.persistence.should_skip = MagicMock(return_value=True)
        self.persistence.save_result = MagicMock()
        
        # Create the builder with persistence
        builder = CoTBuilder(
            search_llm=self.mock_llm,
            search=self.mock_search,
            verifier=self.mock_verifier,
            persistence=self.persistence
        )
        
        question = "What is 2 + 2?"
        ground_truth = "4"
        
        # Call build
        result = builder.build(question, ground_truth)
        
        # Assert should_skip was called
        self.persistence.should_skip.assert_called_once_with(question, ground_truth)
        
        # Assert search was not called (question was skipped)
        self.mock_search.assert_not_called()
        
        # Assert save_result was not called
        self.persistence.save_result.assert_not_called()
        
        # Assert result is None (skipped)
        self.assertIsNone(result)

    def test_save_results(self):
        """Test that results are saved when build is called."""
        # Mock persistence methods
        self.persistence.should_skip = MagicMock(return_value=False)
        self.persistence.save_result = MagicMock()
        
        # Create the builder with persistence
        builder = CoTBuilder(
            search_llm=self.mock_llm,
            search=self.mock_search,
            verifier=self.mock_verifier,
            persistence=self.persistence
        )
        
        question = "What is 2 + 2?"
        ground_truth = "4"
        
        # Call build
        result = builder.build(question, ground_truth)
        
        # Assert search was called
        self.mock_search.assert_called_once()
        
        # Assert save_result was called
        self.persistence.save_result.assert_called_once_with(
            result=self.mock_search_result,
            question=question,
            ground_truth=ground_truth
        )
        
        # Assert result is the mock result
        self.assertEqual(result, self.mock_search_result)

    def test_batch_processing_with_persistence(self):
        """Test batch processing with persistence."""
        # Mock persistence methods
        self.persistence.should_skip = MagicMock(return_value=False)
        self.persistence.save_result = MagicMock()
        self.persistence.setup_batch_run = MagicMock()
        
        # Create the builder with persistence
        builder = CoTBuilder(
            search_llm=self.mock_llm,
            search=self.mock_search,
            verifier=self.mock_verifier,
            persistence=self.persistence
        )
        
        questions = ["What is 2 + 2?", "What is 3 + 3?"]
        ground_truths = ["4", "6"]
        
        # Call build_batch
        results = builder.build_batch(
            questions=questions,
            ground_truth_answers=ground_truths,
            progress_bar=False
        )
        
        # Assert setup_batch_run was called
        self.persistence.setup_batch_run.assert_called_once_with(len(questions))
        
        # Assert should_skip was called for each question
        self.assertEqual(self.persistence.should_skip.call_count, 2)
        
        # Assert search was called for each question
        self.assertEqual(self.mock_search.call_count, 2)
        
        # Assert save_result was called for each question
        self.assertEqual(self.persistence.save_result.call_count, 2)
        
        # Assert results contain the mock results
        self.assertEqual(len(results), 2)
        for result in results:
            self.assertEqual(result, self.mock_search_result)

    def test_load_processed_results(self):
        """Test loading already processed results when build_batch is called with load_processed=True."""
        # Setup mock result data
        mock_result_data = {
            "id": "test_id",
            "question": "test_question",
            "ground_truth": "test_answer",
            "success": True,
            "timestamp": "2023-01-01T00:00:00",
            "result": {"success": True, "mock": "data"}
        }
        
        # Mock persistence methods
        self.persistence.load_results = MagicMock(return_value=[mock_result_data])
        self.persistence.should_skip = MagicMock(return_value=True)
        self.persistence.save_result = MagicMock()
        self.persistence.setup_batch_run = MagicMock()
        
        # Mock SearchResult.deserialize
        with patch(
            "cot_forge.reasoning.types.SearchResult.deserialize",
            return_value=self.mock_search_result
        ) as mock_deserialize:
            # Create the builder with persistence
            builder = CoTBuilder(
                search_llm=self.mock_llm,
                search=self.mock_search,
                verifier=self.mock_verifier,
                persistence=self.persistence,
                strategy_registry=self.mock_strategy_registry
            )
            
            questions = ["What is 2 + 2?"]
            ground_truths = ["4"]
            
            # Call build_batch with load_processed=True
            results = builder.build_batch(
                questions=questions,
                ground_truth_answers=ground_truths,
                load_processed=True,
                progress_bar=False
            )
            
            # Assert load_results was called
            self.persistence.load_results.assert_called_once()
            
            # Assert SearchResult.deserialize was called with the result data
            mock_deserialize.assert_called_once_with(
                mock_result_data["result"],
                self.mock_strategy_registry
            )
            
            # Assert results contain the loaded result
            self.assertEqual(len(results), 1)
            self.assertEqual(results[0], self.mock_search_result)

    def test_multi_threaded_batch_with_persistence(self):
        """Test multi-threaded batch processing with persistence."""
        # Mock persistence methods
        self.persistence.should_skip = MagicMock(return_value=False)
        self.persistence.save_result = MagicMock()
        self.persistence.setup_batch_run = MagicMock()
        
        # Create the builder with persistence
        builder = CoTBuilder(
            search_llm=self.mock_llm,
            search=self.mock_search,
            verifier=self.mock_verifier,
            persistence=self.persistence
        )
        
        questions = ["What is 2 + 2?", "What is 3 + 3?", "What is 4 + 4?"]
        ground_truths = ["4", "6", "8"]
        
        # Call build_batch with multi_thread=True
        results = builder.build_batch(
            questions=questions,
            ground_truth_answers=ground_truths,
            multi_thread=True,
            max_workers=2,
            progress_bar=False
        )
        
        # Assert setup_batch_run was called
        self.persistence.setup_batch_run.assert_called_once_with(len(questions))
        
        # Assert should_skip was called for each question
        self.assertEqual(self.persistence.should_skip.call_count, 3)
        
        # Assert search was called for each question
        self.assertEqual(self.mock_search.call_count, 3)
        
        # Assert save_result was called for each question
        self.assertEqual(self.persistence.save_result.call_count, 3)
        
        # Assert results contain the mock results
        self.assertEqual(len(results), 3)
        for result in results:
            self.assertEqual(result, self.mock_search_result)

    def test_integration_with_real_persistence(self):
        """Test an end-to-end integration with a real persistence manager."""
        # Create the builder with real persistence
        builder = CoTBuilder(
            search_llm=self.mock_llm,
            search=self.mock_search,
            verifier=self.mock_verifier,
            persistence=self.persistence
        )
        
        question = "What is 2 + 2?"
        ground_truth = "4"
        
        # Call build to save a result
        builder.build(question, ground_truth)
        
        # Assert the config file exists
        config_path = self.persistence.config_path
        self.assertTrue(config_path.exists())
        
        # Assert the metadata file exists
        metadata_path = self.persistence.metadata_path
        self.assertTrue(metadata_path.exists())
        
        # Assert the results file exists
        results_path = self.persistence.results_path
        self.assertTrue(results_path.exists())
        
        # Load the metadata file and check values
        with open(metadata_path) as f:
            metadata = json.load(f)
            self.assertEqual(metadata["dataset_name"], self.dataset_name)
            self.assertEqual(metadata["search_name"], self.mock_search.name)
            self.assertEqual(metadata["completed_items"], 1)
            self.assertEqual(metadata["successful_items"], 1)  # Mock result has success=True
            self.assertEqual(len(metadata["processed_ids"]), 1)
        
        # Reset the mock to verify the second call
        self.mock_search.reset_mock()
        
        # Call build again with the same question (should skip)
        result2 = builder.build(question, ground_truth)
        
        # Assert search was not called (question was skipped)
        self.mock_search.assert_not_called()
        
        # Assert result is None (skipped)
        self.assertIsNone(result2)

    def test_auto_resume(self):
        """Test auto-resuming from a previous state."""
        # Test with auto_resume=True
        with patch.object(PersistenceManager, 'load_metadata', return_value=True) as mock_load_metadata:
            PersistenceManager(
                dataset_name=self.dataset_name,
                search_name=self.mock_search.name,
                base_dir=self.temp_dir.name,
                auto_resume=True
            )
            
            # Assert load_metadata was called
            mock_load_metadata.assert_called_once()
        
        # Test with auto_resume=False
        with patch.object(PersistenceManager, 'load_metadata', return_value=True) as mock_load_metadata:
            PersistenceManager(
                dataset_name=self.dataset_name,
                search_name=self.mock_search.name,
                base_dir=self.temp_dir.name,
                auto_resume=False
            )
            
            # Assert load_metadata was NOT called
            mock_load_metadata.assert_not_called()


if __name__ == '__main__':
    unittest.main()
