import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from cot_forge.llm import LLMProvider
from cot_forge.persistence import PersistenceManager
from cot_forge.post_processing.reasoning_processor import ReasoningProcessor
from cot_forge.reasoning.strategies import StrategyRegistry


class TestReasoningProcessor(unittest.TestCase):
  """Tests for the ReasoningProcessor class."""

  def setUp(self):
    """Set up mock dependencies for ReasoningProcessor."""
    self.mock_llm_provider = Mock(spec=LLMProvider)
    self.dataset_name = "test_dataset"
    self.search_name = "test_search"
    self.mock_persistence = Mock(spec=PersistenceManager)

    # Configure the mock with search_dir attribute
    self.mock_persistence.search_dir = Path(
        "test_base/test_dataset/test_search")

    # Configure load_config to return a mock strategy registry
    mock_config = {"strategy_registry": {"some": "config"}}
    self.mock_persistence.load_config.return_value = mock_config

    # Patch PersistenceManager so we don't create real directories
    patcher = patch(
        'cot_forge.post_processing.reasoning_processor.PersistenceManager',
        return_value=self.mock_persistence
    )
    self.addCleanup(patcher.stop)
    self.mock_persistence_class = patcher.start()

  def test_init_basic(self):
    """Test basic initialization of ReasoningProcessor."""
    processor = ReasoningProcessor(
        llm_provider=self.mock_llm_provider,
        dataset_name=self.dataset_name,
        search_name=self.search_name,
        base_dir="test_base"
    )
    self.assertEqual(processor.llm_provider, self.mock_llm_provider)
    self.assertEqual(processor.dataset_name, self.dataset_name)
    self.assertEqual(processor.search_name, self.search_name)
    self.mock_persistence_class.assert_called_once_with(
        dataset_name=self.dataset_name,
        search_name=self.search_name,
        base_dir="test_base"
    )

  def test_get_strategy_registry(self):
    """Test that get_strategy_registry returns a StrategyRegistry."""
    processor = ReasoningProcessor(
        llm_provider=self.mock_llm_provider,
        dataset_name=self.dataset_name,
        search_name=self.search_name
    )
    registry = processor.get_strategy_registry()
    self.assertIsInstance(registry, StrategyRegistry)

  @unittest.skip("Enable after implementing line-by-line writing feature.")
  def test_line_by_line_writing(self):
    """Test that processor writes results line-by-line to support large files."""
    # Once implemented, mock the file writing and check calls to write lines
    pass


if __name__ == '__main__':
  unittest.main()
