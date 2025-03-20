import json
import logging
from unittest.mock import MagicMock, patch

import pytest

from cot_forge.llm import LLMProvider
from cot_forge.reasoning.strategies import Strategy
from cot_forge.reasoning.types import ReasoningNode
from cot_forge.reasoning.verifiers.base import BaseVerifier
from cot_forge.reasoning.verifiers.llm_verifiers import LLMJudgeVerifier
from cot_forge.reasoning.verifiers.prompts import DEFAULT_VERIFICATION_PROMPT, VERIFICATION_FORMAT_PROMPT
from cot_forge.utils.parsing import extract_final_answer_from_cot, parse_json_response


class TestLLMJudgeVerifier:
    def setup_method(self):
        # Create a mock LLM provider
        self.mock_llm_provider = MagicMock(spec=LLMProvider)
        
        # Create the verifier
        self.verifier = LLMJudgeVerifier(
            llm_provider=self.mock_llm_provider,
            llm_kwargs={"temperature": 0.0}
        )
        
        # Mock strategy
        self.mock_strategy = MagicMock(spec=Strategy)
        self.mock_strategy.name = "test_strategy"
        
        # Sample reasoning node
        self.node = ReasoningNode(
            strategy=self.mock_strategy,
            prompt="What is 2+2?",
            response="I need to add 2 and 2 together. The answer is 4.",
            cot=[
                {"action": "Inner Thinking", "content": "I need to add 2 and 2 together."},
                {"action": "Final Conclusion", "content": "The answer is 4."}
            ]
        )
        
        # Sample incorrect reasoning node
        self.incorrect_node = ReasoningNode(
            strategy=self.mock_strategy,
            prompt="What is 2+2?",
            response="I think I should multiply. The answer is 5.",
            cot=[
                {"action": "Inner Thinking", "content": "I think I should multiply."},
                {"action": "Final Conclusion", "content": "The answer is 5."}
            ]
        )

    def test_init(self):
        """Test proper initialization of the verifier."""
        assert self.verifier.llm_provider == self.mock_llm_provider
        assert self.verifier.llm_kwargs == {"temperature": 0.0}
        assert self.verifier.prompt_template == DEFAULT_VERIFICATION_PROMPT
        
        # Test with custom prompt template
        custom_template = "Custom template: {final_answer} vs {ground_truth_answer}"
        custom_verifier = LLMJudgeVerifier(
            llm_provider=self.mock_llm_provider,
            prompt_template=custom_template
        )
        assert custom_verifier.prompt_template == custom_template

    def test_build_prompt(self):
        """Test building the verification prompt."""
        prompt = self.verifier.build_prompt(
            final_answer="The answer is 4.",
            ground_truth_answer="4"
        )
        
        # Check that the prompt contains the expected elements
        assert "The answer is 4." in prompt
        assert "4" in prompt
        assert VERIFICATION_FORMAT_PROMPT in prompt

    def test_parse_response_valid(self):
        """Test parsing a valid verification response."""
        # Create a sample LLM response in the expected format
        sample_response = json.dumps({
            "verification": {
                "result": "yes",
                "explanation": "The answer 4 is correct."
            }
        })
        
        result, explanation = self.verifier.parse_response(sample_response)
        assert result == "yes"
        assert explanation == "The answer 4 is correct."

    def test_parse_response_invalid_json(self):
        """Test parsing an invalid JSON response."""
        sample_response = "This is not valid JSON"
        
        result, explanation = self.verifier.parse_response(sample_response)
        assert result is False
        assert "Error:" in explanation

    def test_verify_correct_answer(self):
        """Test verification of a correct answer."""
        # Mock the LLM response
        mock_response = json.dumps({
            "verification": {
                "result": "yes",
                "explanation": "The answer 4 is correct for 2+2."
            }
        })
        self.mock_llm_provider.generate.return_value = mock_response
        
        # Perform verification
        is_correct, explanation = self.verifier.verify(
            node=self.node,
            question="What is 2+2?",
            ground_truth_answer="4"
        )
        
        # Check results
        assert is_correct is True
        assert explanation == "The answer 4 is correct for 2+2."
        
        # Verify the LLM was called with the correct prompt
        self.mock_llm_provider.generate.assert_called_once()
        prompt_arg = self.mock_llm_provider.generate.call_args[1]["prompt"]
        assert "The answer is 4." in prompt_arg
        assert "4" in prompt_arg

    def test_verify_incorrect_answer(self):
        """Test verification of an incorrect answer."""
        # Mock the LLM response
        mock_response = json.dumps({
            "verification": {
                "result": "no",
                "explanation": "The answer 5 is incorrect for 2+2. The correct answer is 4."
            }
        })
        self.mock_llm_provider.generate.return_value = mock_response
        
        # Perform verification
        is_correct, explanation = self.verifier.verify(
            node=self.incorrect_node,
            question="What is 2+2?",
            ground_truth_answer="4"
        )
        
        # Check results
        assert is_correct is False
        assert "incorrect" in explanation

    def test_verify_empty_response(self):
        """Test verification with an empty LLM response."""
        # Mock the LLM response
        self.mock_llm_provider.generate.return_value = ""
        
        # Perform verification
        is_correct, explanation = self.verifier.verify(
            node=self.node,
            question="What is 2+2?",
            ground_truth_answer="4"
        )
        
        # Check results
        assert is_correct is False
        assert "Empty response" in explanation

    def test_verify_llm_error(self):
        """Test verification when LLM raises an error."""
        # Mock the LLM to raise an exception
        self.mock_llm_provider.generate.side_effect = Exception("API error")
        
        # Perform verification
        is_correct, explanation = self.verifier.verify(
            node=self.node,
            question="What is 2+2?",
            ground_truth_answer="4"
        )
        
        # Check results
        assert is_correct is False
        assert "Error:" in explanation
        assert "API error" in explanation

    def test_verify_node_without_cot(self):
        """Test verification of a node without a CoT."""
        # Create a node without CoT
        node_without_cot = ReasoningNode(
            strategy=self.mock_strategy,
            prompt="What is 2+2?",
            response="4",
            cot=None
        )
        
        # Perform verification
        is_correct, explanation = self.verifier.verify(
            node=node_without_cot,
            question="What is 2+2?",
            ground_truth_answer="4"
        )
        
        # Check results
        assert is_correct is False
        assert "Node.cot is None" in explanation

    @patch('cot_forge.reasoning.verifiers.llm_verifiers.extract_final_answer_from_cot')
    def test_verify_no_final_answer(self, mock_extract):
        """Test verification when no Final Conclusion can be extracted."""
        # Mock the extract function to return None
        mock_extract.return_value = None
        
        # Perform verification
        is_correct, explanation = self.verifier.verify(
            node=self.node,
            question="What is 2+2?",
            ground_truth_answer="4"
        )
        
        # Check results
        assert is_correct is False
        assert "No Final Conclusion" in explanation
        
        # Check that the node metadata was updated
        assert self.node.metadata.get("warning") == "missing_final_conclusion"

    def test_callable_interface(self):
        """Test the callable interface of the verifier."""
        # Mock the LLM response
        mock_response = json.dumps({
            "verification": {
                "result": "yes",
                "explanation": "The answer 4 is correct for 2+2."
            }
        })
        self.mock_llm_provider.generate.return_value = mock_response
        
        # Use the verifier as a callable
        is_correct, explanation = self.verifier(
            node=self.node,
            question="What is 2+2?",
            ground_truth_answer="4"
        )
        
        # Check results
        assert is_correct is True
        assert explanation == "The answer 4 is correct for 2+2."
        
        # Verify the LLM was called
        self.mock_llm_provider.generate.assert_called_once()


# Create a minimal concrete implementation to test BaseVerifier functionality
class SimpleVerifier(BaseVerifier):
    """Simple verifier implementation for testing the base class."""
    
    def verify(self, node, question, ground_truth_answer, **kwargs):
        if not node.cot:
            return False, "No CoT available"
        
        final_answer = extract_final_answer_from_cot(node.cot)
        if not final_answer:
            return False, "No Final Conclusion found"
        
        is_correct = ground_truth_answer.lower() in final_answer.lower()

        explanation = "Answer is correct" if is_correct else "Answer is incorrect"
        return is_correct, explanation


class TestBaseVerifier:
    """Test the functionality of the BaseVerifier abstract base class."""
    
    def setup_method(self):
        # Create a concrete verifier instance
        self.verifier = SimpleVerifier(
            name="simple_verifier",
            description="A simple exact match verifier for testing"
        )
        
        # Mock strategy
        self.mock_strategy = MagicMock(spec=Strategy)
        self.mock_strategy.name = "test_strategy"
        
        # Sample reasoning node
        self.node = ReasoningNode(
            strategy=self.mock_strategy,
            prompt="What is 2+2?",
            response="I need to add 2 and 2 together. The answer is 4.",
            cot=[
                {"action": "Inner Thinking", "content": "I need to add 2 and 2 together."},
                {"action": "Final Conclusion", "content": "The answer is 4."}
            ]
        )
    
    def test_init(self):
        """Test proper initialization of the base verifier."""
        assert self.verifier.name == "simple_verifier"
        assert self.verifier.description == "A simple exact match verifier for testing"
        assert self.verifier.llm_provider is None
        assert self.verifier.llm_kwargs == {}
        
        # Test with LLM provider
        mock_llm = MagicMock(spec=LLMProvider)
        verifier_with_llm = SimpleVerifier(
            name="with_llm",
            description="Verifier with LLM",
            llm_provider=mock_llm,
            llm_kwargs={"temperature": 0.5}
        )
        assert verifier_with_llm.llm_provider == mock_llm
        assert verifier_with_llm.llm_kwargs == {"temperature": 0.5}
    
    def test_verify(self):
        """Test the verify method of the concrete implementation."""
        is_correct, explanation = self.verifier.verify(
            node=self.node,
            question="What is 2+2?",
            ground_truth_answer="4"
        )
        assert is_correct is True
        assert explanation == "Answer is correct"
        
        # Test with incorrect answer
        is_correct, explanation = self.verifier.verify(
            node=self.node,
            question="What is 2+2?",
            ground_truth_answer="5"
        )
        assert is_correct is False
        assert explanation == "Answer is incorrect"
    
    def test_callable_interface(self):
        """Test the callable interface."""
        is_correct, explanation = self.verifier(
            node=self.node,
            question="What is 2+2?",
            ground_truth_answer="4"
        )
        assert is_correct is True
        assert explanation == "Answer is correct"