import json
from unittest.mock import MagicMock, patch

import pytest

from cot_forge.llm import LLMProvider
from cot_forge.reasoning.scorers.llm_scorers import LLMScorerBase, ProbabilityFinalAnswerScorer


class TestLLMScorerBase:
    def setup_method(self):
        # Create a mock LLM provider
        self.mock_llm_provider = MagicMock(spec=LLMProvider)
        
        # Create a simple concrete subclass for testing the abstract base class
        class ConcreteLLMScorer(LLMScorerBase):
            def score(self, cot_list, question, ground_truth_answer, **kwargs):
                return {"strategy_1": 0.9, "strategy_2": 0.5}
        
        self.scorer = ConcreteLLMScorer(
            name="test_scorer",
            description="A test scorer",
            llm_provider=self.mock_llm_provider,
            llm_kwargs={"temperature": 0.0}
        )
    
    def test_generate_and_parse_scores_success(self):
        # Arrange
        mock_response = json.dumps({"scoring": {"strategy_1": 0.9, "strategy_2": 0.5}})
        self.mock_llm_provider.generate.return_value = mock_response
        
        # Act
        result, error_msg = self.scorer.generate_and_parse_scores(
            prompt="Test prompt",
            on_error="retry"
        )
        
        # Assert
        assert result == {"strategy_1": 0.9, "strategy_2": 0.5}
        assert error_msg is None
        self.mock_llm_provider.generate.assert_called_once_with(
            prompt="Test prompt",
            temperature=0.0
        )
    
    def test_generate_and_parse_scores_llm_error_retry(self):
        # Arrange
        self.mock_llm_provider.generate.side_effect = [
            Exception("LLM error"),  # First call fails
            json.dumps({"scoring": {"strategy_1": 0.9, "strategy_2": 0.5}})  # Second call succeeds
        ]
        
        # Act
        result, error_msg = self.scorer.generate_and_parse_scores(
            prompt="Test prompt",
            on_error="retry",
            max_retries=2
        )
        
        # Assert
        assert result == {"strategy_1": 0.9, "strategy_2": 0.5}
        assert error_msg is None
        assert self.mock_llm_provider.generate.call_count == 2
    
    def test_generate_and_parse_scores_parsing_error_retry(self):
        # Arrange
        self.mock_llm_provider.generate.side_effect = [
            "Invalid JSON",  # First call returns unparseable response
            json.dumps({"scoring": {"strategy_1": 0.9, "strategy_2": 0.5}})  # Second call succeeds
        ]
        
        # Act
        result, error_msg = self.scorer.generate_and_parse_scores(
            prompt="Test prompt",
            on_error="retry",
            max_retries=2
        )
        
        # Assert
        assert result == {"strategy_1": 0.9, "strategy_2": 0.5}
        assert error_msg is None
        assert self.mock_llm_provider.generate.call_count == 2
    
    def test_generate_and_parse_scores_max_retries_exceeded(self):
        # Arrange
        self.mock_llm_provider.generate.side_effect = Exception("LLM error")
        
        # Act & Assert
        with pytest.raises(RuntimeError, match="LLM generation for scoring failed"):
            self.scorer.generate_and_parse_scores(
                prompt="Test prompt",
                on_error="retry",
                max_retries=2
            )
        
        assert self.mock_llm_provider.generate.call_count == 3  # Initial + 2 retries
    
    def test_generate_and_parse_scores_raise_immediately(self):
        # Arrange
        self.mock_llm_provider.generate.side_effect = Exception("LLM error")
        
        # Act & Assert
        with pytest.raises(RuntimeError, match="LLM generation for scoring failed"):
            self.scorer.generate_and_parse_scores(
                prompt="Test prompt",
                on_error="raise"
            )
        
        assert self.mock_llm_provider.generate.call_count == 1  # No retries
    
    def test_generate_and_parse_scores_continue_on_error(self):
        # Arrange
        self.mock_llm_provider.generate.side_effect = Exception("LLM error")
        
        # Act
        result, error_msg = self.scorer.generate_and_parse_scores(
            prompt="Test prompt",
            on_error="continue"
        )
        
        # Assert
        assert result is None
        assert "LLM error" in error_msg
        assert self.mock_llm_provider.generate.call_count == 1  # No retries


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
    def test_score_success(self, mock_extract):
        # Arrange
        mock_extract.side_effect = ["42", "24"]  # Return values for each call
        
        # Mock the generate_and_parse_scores method
        with patch.object(
            self.scorer, 
            'generate_and_parse_scores', 
            return_value=({"strategy_1": 0.9, "strategy_2": 0.5}, None)
        ) as mock_generate:
            
            # Act
            result = self.scorer.score(
                cot_list=self.cot_list,
                question="What is the answer?",
                ground_truth_answer="42"
            )
            
            # Assert
            assert result == {"strategy_1": 0.9, "strategy_2": 0.5}
            mock_extract.assert_called()
            mock_generate.assert_called_once()
            
            # Check that the prompt was built correctly
            prompt_arg = mock_generate.call_args[1]["prompt"]
            assert "What is the answer?" in prompt_arg
            assert "42" in prompt_arg  # Ground truth
            assert "strategy_1: 42" in prompt_arg
            assert "strategy_2: 24" in prompt_arg
    
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
    def test_score_llm_error(self, mock_extract):
        # Arrange
        mock_extract.side_effect = ["42", "24"]
        
        # Mock generate_and_parse_scores to raise an error
        with patch.object(
            self.scorer, 
            'generate_and_parse_scores', 
            side_effect=RuntimeError("LLM error")
        ) as mock_generate:
            
            # Act
            with pytest.raises(RuntimeError, match="LLM error"):
                self.scorer.score(
                    cot_list=self.cot_list,
                    question="What is the answer?",
                    ground_truth_answer="42"
                )
            
            # Assert
            mock_extract.assert_called()
            mock_generate.assert_called_once()
    
    @patch('cot_forge.reasoning.scorers.llm_scorers.extract_final_answer_from_cot')
    def test_score_with_warning(self, mock_extract):
        # Arrange
        mock_extract.side_effect = ["42", "24"]
        
        # Mock generate_and_parse_scores to return a result with a warning
        with patch.object(
            self.scorer, 
            'generate_and_parse_scores', 
            return_value=({"strategy_1": 0.9, "strategy_2": 0.5}, "Some warning")
        ) as mock_generate:
            
            # Act
            result = self.scorer.score(
                cot_list=self.cot_list,
                question="What is the answer?",
                ground_truth_answer="42"
            )
            
            # Assert
            assert result == {"strategy_1": 0.9, "strategy_2": 0.5}
            mock_extract.assert_called()
            mock_generate.assert_called_once()
    
    @patch('cot_forge.reasoning.scorers.llm_scorers.ScorerPromptTemplate')
    @patch('cot_forge.reasoning.scorers.llm_scorers.extract_final_answer_from_cot')
    def test_full_integration(self, mock_extract, mock_template):
        # Arrange
        mock_extract.side_effect = ["42", "24"]
        mock_template.build_prompt.return_value = "test prompt"
        
        mock_response = json.dumps({"scoring": {"strategy_1": "0.9", "strategy_2": "0.5"}})
        self.mock_llm_provider.generate.return_value = mock_response
        
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
        self.mock_llm_provider.generate.assert_called_once()