from unittest.mock import MagicMock, patch

import pytest

from cot_forge.llm import LLMProvider
from cot_forge.reasoning.scorers.llm_scorers import ProbabilityFinalAnswerScorer


class TestProbabilityFinalAnswerScorer:
    def setup_method(self):
        # Create a mock LLM provider
        self.mock_llm_provider = MagicMock(spec=LLMProvider)
        
        # Create the scorer
        self.scorer = ProbabilityFinalAnswerScorer(
            llm_provider=self.mock_llm_provider,
            llm_kwargs={"temperature": 0.0}
        )
        
        # Sample CoT list for testing
        self.cot_list = [
            {
                "strategy_name": "strategy_1",
                "cot": [
                    {"action": "Inner Thinking", "content": "Let me think..."},
                    {"action": "Final Answer", "content": "The answer is 42."}
                ]
            },
            {
                "strategy_name": "strategy_2",
                "cot": [
                    {"action": "Inner Thinking", "content": "I need to calculate..."},
                    {"action": "Final Answer", "content": "The answer is 24."}
                ]
            }
        ]
    
    @patch('cot_forge.reasoning.scorers.llm_scorers.extract_final_answer_from_cot')
    @patch('cot_forge.utils.search_utils.execute_with_fallback')
    def test_score_success(self, mock_execute, mock_extract):
        # Arrange
        mock_extract.side_effect = ["42", "24"]  # Return values for each call
        mock_execute.return_value = (("response text", {"strategy_1": 0.9, "strategy_2": 0.5}), None)
        
        # Act
        result = self.scorer.score(
            cot_list=self.cot_list,
            question="What is the answer?",
            ground_truth_answer="42"
        )
        
        # Assert
        assert result == {"strategy_1": 0.9, "strategy_2": 0.5}
        mock_extract.assert_called()
        mock_execute.assert_called_once()
    
    @patch('cot_forge.reasoning.scorers.llm_scorers.extract_final_answer_from_cot')
    def test_score_extraction_error(self, mock_extract):
        # Arrange
        mock_extract.side_effect = Exception("Extraction error")
        
        # Act
        result = self.scorer.score(
            cot_list=self.cot_list,
            question="What is the answer?",
            ground_truth_answer="42"
        )
        
        # Assert
        assert result == {}  # Empty dict returned on error
        mock_extract.assert_called_once()  # Only called once because it raises an exception
    
    @patch('cot_forge.reasoning.scorers.llm_scorers.extract_final_answer_from_cot')
    @patch('cot_forge.utils.search_utils.execute_with_fallback')
    def test_score_llm_error(self, mock_execute, mock_extract):
        # Arrange
        mock_extract.side_effect = ["42", "24"]
        mock_execute.side_effect = RuntimeError("LLM generation and json parsing failed")
        
        # Act
        with pytest.raises(RuntimeError, match="LLM generation and json parsing failed"):
            self.scorer.score(
                cot_list=self.cot_list,
                question="What is the answer?",
                ground_truth_answer="42"
            )
        
        # Assert
        mock_extract.assert_called()
        mock_execute.assert_called_once()
    
    @patch('cot_forge.reasoning.scorers.llm_scorers.ScorerPromptTemplate')
    @patch('cot_forge.reasoning.scorers.llm_scorers.extract_final_answer_from_cot')
    @patch('cot_forge.utils.search_utils.execute_with_fallback')
    def test_full_integration(self, mock_execute, mock_extract, mock_template):
        # Arrange
        mock_extract.side_effect = ["42", "24"]
        mock_template.build_prompt.return_value = "test prompt"
        mock_execute.return_value = (("response text", {"strategy_1": 0.9, "strategy_2": 0.5}), None)
        
        # Act
        result = self.scorer.score(
            cot_list=self.cot_list,
            question="What is the answer?",
            ground_truth_answer="42"
        )
        
        # Assert
        assert result == {"strategy_1": 0.9, "strategy_2": 0.5}
        mock_extract.assert_called()
        mock_template.build_prompt.assert_called_once()
        mock_execute.assert_called_once()