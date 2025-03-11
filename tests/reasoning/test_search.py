import unittest
from unittest.mock import patch

from cot_forge.reasoning import naive_linear_search
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
                },
                {
                    "action": "Verification",
                    "content": "The answer is correct because 2 + 2 = 4."
                }
            ]
        }"""
        mock_verifier.verify.return_value = True

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
        mock_verifier.verify.assert_called_once()
        # Check that the node was passed to verifier
        verify_args = mock_verifier.verify.call_args
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
        mock_verifier.verify.return_value = False

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
                    },
                    {
                        "action": "Verification",
                        "content": "The answer is correct because 2 + 2 = 5."
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
                    },
                    {
                        "action": "Verification",
                        "content": "The answer is correct because 2 + 2 = 4."
                    }
                ]
            }"""
        ]
        mock_llm_provider.generate.side_effect = mock_responses
        mock_verifier.verify.side_effect = [False, True]

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
        self.assertEqual(mock_verifier.verify.call_count, 2)
        
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
        mock_verifier.verify.return_value = False

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
        self.assertEqual(mock_verifier.verify.call_count, 3)
        
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
        
        print('HOWDY', result)

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
            for node in result['final_node'].get_full_chain()
        )
        self.assertIsNone(result['final_answer'])
        # Chain should have gone to max depth
        assert result['metadata']['depth'] == max_depth
        
            
            
            
            
