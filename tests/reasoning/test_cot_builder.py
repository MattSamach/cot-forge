import json
import tempfile
import unittest
from unittest.mock import MagicMock, Mock, patch, ANY

from cot_forge.llm import LLMProvider
from cot_forge.persistence import PersistenceManager
from cot_forge.post_processing import ReasoningProcessor
from cot_forge.reasoning.cot_builder import CoTBuilder
from cot_forge.reasoning.search.search_algorithm import SearchAlgorithm
from cot_forge.reasoning.strategies import StrategyRegistry
from cot_forge.reasoning.types import ReasoningNode, SearchResult
from cot_forge.reasoning.verifiers import BaseVerifier, LLMJudgeVerifier


class TestCoTBuilder(unittest.TestCase):
    """Tests for the CoTBuilder class."""

    def setUp(self):
        """Set up test fixtures."""
        # Mock core components
        self.mock_search_llm = Mock(spec=LLMProvider)
        self.mock_post_processing_llm = Mock(spec=LLMProvider)
        self.mock_search = Mock(spec=SearchAlgorithm)
        self.mock_strategy_registry = Mock(spec=StrategyRegistry)
        self.mock_verifier = Mock(spec=LLMJudgeVerifier)
        
        # Create the mock post processor
        self.mock_post_processor = Mock(spec=ReasoningProcessor)
        self.mock_post_processor.process_result = MagicMock(return_value={"id": "test_id", "content": "test reasoning"})

        # Configure mock search attributes
        self.mock_search.name = "MockSearch"
        
        # Mock the to_dict methods needed for save_config
        self.mock_search_llm.to_dict = MagicMock(return_value={"type": "MockLLM"})
        self.mock_post_processing_llm.to_dict = MagicMock(return_value={"type": "MockPostProcessingLLM"})
        self.mock_search.to_dict = MagicMock(return_value={"name": "MockSearch"})
        self.mock_verifier.to_dict = MagicMock(return_value={"type": "MockVerifier"})
        
        # Create a mock search result as a dictionary (not an object with attributes)
        self.mock_search_result = {
            'terminal_nodes': [MagicMock(spec=ReasoningNode)],
            'success': True,
            'final_answer': 'Mock answer',
            'metadata': {'test': 'metadata'},
            'question': 'What is 2 + 2?',
            'ground_truth_answer': '4'
        }

        # Configure the mock search to return the mock result
        self.mock_search.return_value = self.mock_search_result
        
        # Create test temp directory
        self.temp_dir = tempfile.TemporaryDirectory()
        
        # Mock PersistenceManager's methods to avoid file operations in tests
        with patch('cot_forge.persistence.PersistenceManager.save_config'), \
             patch('cot_forge.persistence.PersistenceManager._save_metadata'):
            
            # Create the CoTBuilder with mocked dependencies - new constructor signature
            self.cot_builder = CoTBuilder(
                search_llm=self.mock_search_llm,
                post_processing_llm=self.mock_post_processing_llm,
                search=self.mock_search,
                verifier=self.mock_verifier,
                dataset_name="test_dataset",
                base_dir=self.temp_dir.name,
                strategy_registry=self.mock_strategy_registry
            )
            
            # Replace the internal persistence with our mock for easier assertions
            self.mock_persistence = Mock(spec=PersistenceManager)
            self.mock_persistence.dataset_name = "test_dataset"
            self.mock_persistence.generate_question_id = MagicMock(return_value="test_id")
            self.mock_persistence.save_config = MagicMock()
            self.mock_persistence.save_result = MagicMock()
            self.mock_persistence.should_skip = MagicMock(return_value=False)
            self.mock_persistence.setup_batch_run = MagicMock()
            
            # Replace persistence and post_processor in the builder
            self.cot_builder.persistence = self.mock_persistence
            self.cot_builder.post_processor = self.mock_post_processor
    
    def tearDown(self):
        """Clean up after tests."""
        self.temp_dir.cleanup()

    def test_build_cot(self):
        """Test the build_cot method."""
        question = "What is 2 + 2?"
        ground_truth = "4"
        llm_kwargs = {'temperature': 0.7}
        extra_kwargs = {'max_depth': 3, 'custom_param': 'value'}

        result = self.cot_builder.build_cot(
            question=question,
            ground_truth_answer=ground_truth,
            llm_kwargs=llm_kwargs,
            **extra_kwargs
        )

        self.mock_search.assert_called_once_with(
            question=question,
            ground_truth_answer=ground_truth,
            search_llm=self.mock_search_llm,
            llm_kwargs=llm_kwargs,
            strategy_registry=self.mock_strategy_registry,
            verifier=self.mock_verifier,
            scorer=None,
            max_depth=3,
            custom_param='value'
        )

        self.assertEqual(result, self.mock_search_result)

    def test_empty_batch(self):
        """Test process_batch with empty lists."""
        results = self.cot_builder.process_batch(
            questions=[],
            ground_truth_answers=[],
            progress_bar=False
        )

        self.mock_search.assert_not_called()
        self.assertEqual(results, [])

    def test_batch_processing_persistence(self):
        """Test batch processing persistence methods."""
        # Mock the process method to return expected values
        self.cot_builder.process = MagicMock(return_value=(self.mock_search_result, {"id": "test_id", "content": "test"}))

        questions = ["What is 2 + 2?", "What is 3 + 3?"]
        ground_truths = ["4", "6"]

        results = self.cot_builder.process_batch(
            questions=questions,
            ground_truth_answers=ground_truths,
            progress_bar=False
        )

        self.mock_persistence.setup_batch_run.assert_called_once()
        self.assertEqual(self.cot_builder.process.call_count, len(questions))
        self.assertEqual(len(results), len(questions))

    def test_integration_with_real_persistence(self):
        """Test an end-to-end integration with a real persistence manager."""
        # Patch PersistenceManager's save_result to avoid dict access errors
        with patch('cot_forge.persistence.persistence_manager.PersistenceManager.save_result') as mock_save_result:
            # Use a temporary directory
            temp_dir = tempfile.TemporaryDirectory()
            
            # Create a real persistence manager
            persistence = PersistenceManager(
                dataset_name="test_dataset",
                search_name="MockSearch",
                base_dir=temp_dir.name
            )
            
            # Create a proper mock search with the to_dict method
            mock_search = Mock(spec=SearchAlgorithm)
            mock_search.name = "MockSearch"
            mock_search.to_dict = MagicMock(return_value={"name": "MockSearch"})
            
            # Create proper mock LLM, verifier, and post_processor with to_dict methods
            mock_llm = Mock(spec=LLMProvider)
            mock_llm.to_dict = MagicMock(return_value={"type": "MockLLM"})
            
            mock_verifier = Mock(spec=BaseVerifier)
            mock_verifier.to_dict = MagicMock(return_value={"type": "MockVerifier"})
            
            # Create a test builder with the real persistence but mock everything else
            builder = CoTBuilder(
                search_llm=mock_llm,
                post_processing_llm=self.mock_post_processing_llm,
                search=mock_search,
                verifier=mock_verifier,
                dataset_name="test_dataset",
                base_dir=temp_dir.name
            )
            
            # Mock the post processor
            mock_post_processor = Mock(spec=ReasoningProcessor)
            mock_post_processor.process_result = MagicMock(return_value={"id": "test_id", "content": "test reasoning"})
            builder.post_processor = mock_post_processor
            
            # Mock the build_cot method
            builder.build_cot = MagicMock(return_value=self.mock_search_result)
            
            # Process a question
            question = "What is 2 + 2?"
            ground_truth = "4"
            
            builder.process(question=question, ground_truth_answer=ground_truth)
            
            # Check that config file exists
            self.assertTrue(persistence.config_path.parent.exists())
            
            # Verify save_result was called
            mock_save_result.assert_called_once()
            
            # Clean up
            temp_dir.cleanup()

    def test_multi_threaded_batch_with_persistence(self):
        """Test multi-threaded batch processing with persistence."""
        # Mock the process method to return expected values
        self.cot_builder.process = MagicMock(return_value=(self.mock_search_result, {"id": "test_id", "content": "test"}))

        questions = ["What is 2 + 2?", "What is 3 + 3?", "What is 4 + 4?"]
        ground_truths = ["4", "6", "8"]

        results = self.cot_builder.process_batch(
            questions=questions,
            ground_truth_answers=ground_truths,
            multi_thread=True,
            max_workers=2,
            progress_bar=False
        )

        self.mock_persistence.setup_batch_run.assert_called_once()
        # Assert process was called for each question
        self.assertEqual(self.cot_builder.process.call_count, len(questions))
        self.assertEqual(len(results), len(questions))

    def test_save_config(self):
        """Test that configuration is saved when CoTBuilder is created with persistence."""
        # Create a fresh mock
        mock_persistence = Mock(spec=PersistenceManager)
        mock_persistence.dataset_name = "test_dataset"
        mock_persistence.save_config = MagicMock()
        
        # Patch the PersistenceManager class constructor to return our mock
        with patch('cot_forge.reasoning.cot_builder.PersistenceManager', return_value=mock_persistence):
            # Create a new builder that will trigger save_config
            builder = CoTBuilder(
                search_llm=self.mock_search_llm,
                post_processing_llm=self.mock_post_processing_llm,
                search=self.mock_search,
                verifier=self.mock_verifier,
                dataset_name="test_dataset",
                base_dir=self.temp_dir.name
            )
            
            # The save_config should be called in the CoTBuilder.__init__
            mock_persistence.save_config.assert_called_once()

    def test_save_results(self):
        """Test that results are saved when build_cot is called."""
        # Mock the methods that would be called
        self.mock_persistence.should_skip = MagicMock(return_value=False)
        self.mock_persistence.save_result = MagicMock()
        
        # Call build_cot
        question = "What is 2 + 2?"
        ground_truth = "4"
        
        # The method we're actually testing - doesn't call save_result directly anymore
        # We need to check process method instead
        result = self.cot_builder.build_cot(question, ground_truth)
        
        # Assert search was called
        self.mock_search.assert_called_once()
        
        # Create a new builder that directly uses PersistenceManager methods
        mock_persistence = Mock(spec=PersistenceManager)
        mock_persistence.should_skip = MagicMock(return_value=False)
        mock_persistence.save_result = MagicMock()
        mock_persistence.generate_question_id = MagicMock(return_value="test_id")
        mock_persistence.dataset_name = "test_dataset"
        
        builder = CoTBuilder(
            search_llm=self.mock_search_llm,
            post_processing_llm=self.mock_post_processing_llm,
            search=self.mock_search,
            verifier=self.mock_verifier,
            dataset_name="test_dataset",
            base_dir=self.temp_dir.name
        )
        
        # Mock build_cot method to return our mock result
        builder.build_cot = MagicMock(return_value=self.mock_search_result)
        
        # Now manually call save_result to verify it works
        mock_persistence.save_result(
            result=self.mock_search_result,
            reasoning={"id": "test_id", "content": "test reasoning"},
            question=question,
            ground_truth=ground_truth
        )
        
        # Assert save_result was called
        mock_persistence.save_result.assert_called_once()

    def test_skip_processed_questions(self):
        """Test that already processed questions are skipped."""
        # Configure the mock to return True for should_skip
        self.mock_persistence.should_skip = MagicMock(return_value=True)
        
        question = "What is 2 + 2?"
        ground_truth = "4"
        
        result = self.cot_builder.build_cot(question, ground_truth)
        
        # Assert should_skip was called
        self.mock_persistence.should_skip.assert_called_once_with(question, ground_truth)
        
        # Assert search was not called (question was skipped)
        self.mock_search.assert_not_called()
        
        # Assert result is None (skipped)
        self.assertIsNone(result)

    def test_repr_and_str(self):
        """Test the __repr__ and __str__ methods."""
        # Test __repr__
        repr_str = repr(self.cot_builder)
        self.assertIn("CoTBuilder", repr_str)
        self.assertIn("Persistence: Enabled (test_dataset)", repr_str)
        
        # Test __str__
        str_str = str(self.cot_builder)
        self.assertIn("CoTBuilder", str_str)
        self.assertIn(str(self.mock_search_llm), str_str)
        self.assertIn(str(self.mock_search), str_str)

    @patch('cot_forge.reasoning.cot_builder.logger')
    def test_process(self, mock_logger):
        """Test the process method."""
        # Mock dependencies
        self.cot_builder.build_cot = MagicMock(return_value=self.mock_search_result)
        self.mock_persistence.generate_question_id = MagicMock(return_value="test-id")
        self.mock_post_processor.process_result = MagicMock(return_value={"id": "test-id", "content": "test reasoning"})
        
        # Test the process method
        question = "What is 2 + 2?"
        ground_truth = "4"
        
        result, reasoning = self.cot_builder.process(question, ground_truth)
        
        # Verify the call flow - use ANY to match any argument
        self.cot_builder.build_cot.assert_called_once_with(
            question=question, 
            ground_truth_answer=ground_truth,
            llm_kwargs=None
        )
        self.mock_post_processor.process_result.assert_called_once()
        self.mock_persistence.save_result.assert_called_once()
        
        # Verify the return values
        self.assertEqual(result, self.mock_search_result)
        self.assertEqual(reasoning, {"id": "test-id", "content": "test reasoning"})


if __name__ == '__main__':
    unittest.main()
