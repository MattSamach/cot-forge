import unittest
from typing import Optional
from unittest.mock import MagicMock, patch

from src.cot_forge.llm import GeminiLLMProvider, LLMProvider


class TestLLMProviderInterface(unittest.TestCase):
    """Test the LLM Provider interface requirements."""
    
    def test_abstract_methods(self):
        """Verify that concrete subclasses must implement abstract methods."""
        # Should not be able to instantiate the abstract base class
        with self.assertRaises(TypeError):
            LLMProvider(model_name="test-model")
        
        # Create a minimal concrete subclass missing required methods
        class IncompleteProvider(LLMProvider):
            pass
        
        # Should raise TypeError for incomplete implementation
        with self.assertRaises(TypeError):
            IncompleteProvider()
        
        # Create a minimal working concrete subclass
        class MinimalProvider(LLMProvider):
            def generate_completion(self, 
                                    prompt: str,
                                    system_prompt: Optional[str] = None,
                                    temperature: float = 0.7,
                                    max_tokens: Optional[int] = None,
                                    **kwargs) -> str:
                return "Test response"
                
        # Should instantiate successfully
        provider = MinimalProvider(model_name="test-model")
        self.assertEqual(provider.generate("test"), "Test response")
    
    def test_generate_batch_not_implemented(self):
        """Verify generate_batch raises NotImplementedError by default."""
        class MinimalProvider(LLMProvider):
            def generate_completion(self, prompt: str, system_prompt: Optional[str] = None, 
                        temperature: float = 0.7, max_tokens: Optional[int] = None, **kwargs) -> str:
                return "Test response"
        
        provider = MinimalProvider(model_name="test-model")
        with self.assertRaises(NotImplementedError):
            provider.generate_batch(["test"])


@patch('google.genai.Client')
@patch('google.genai.types')
class TestGeminiLLMProvider(unittest.TestCase):
    """Test the Gemini LLM Provider implementation."""
    
    def test_initialization(self, mock_types, mock_client):
        """Test provider initialization sets up client and configuration correctly."""
        provider = GeminiLLMProvider(api_key="fake_key", model_name="test-model")
        
        # Check client was created with API key
        mock_client.assert_called_once_with(api_key="fake_key")
        
        # Verify attributes were set correctly
        self.assertEqual(provider.model_name, "test-model")
        self.assertEqual(provider.client, mock_client.return_value)
        self.assertEqual(provider.types, mock_types)
        
    def test_import_error_handling(self, mock_types, mock_client):
        """Test appropriate error when Google libraries aren't available."""
        with patch('builtins.__import__', side_effect=ImportError):
            with self.assertRaises(ImportError) as context:
                GeminiLLMProvider()
            self.assertIn("Install 'google-genai'", str(context.exception))

    def test_generate_call_structure(self, mock_types, mock_client):
        """Test the generate method calls the API with correct parameters."""
        # Setup mock response
        mock_response = MagicMock()
        mock_response.text = "Generated text"
        mock_client.return_value.models.generate_content.return_value = mock_response
        
        # Create provider and call generate
        provider = GeminiLLMProvider(api_key="fake_key")
        result = provider.generate(
            prompt="Test prompt", 
            system_prompt="System instructions",
            temperature=0.5,
            max_tokens=100
        )
        
        # Verify API was called correctly
        mock_client.return_value.models.generate_content.assert_called_once()
        call_args = mock_client.return_value.models.generate_content.call_args
        
        # Check model parameter
        self.assertEqual(call_args[1]['model'], "gemini-2.0-flash")
        
        # Check contents parameter contains prompt
        self.assertEqual(call_args[1]['contents'], ["Test prompt"])
        
        # Check result is text from response
        self.assertEqual(result, "Generated text")
        
        # Mock types.GenerateContentConfig should be called with correct parameters
        mock_types.GenerateContentConfig.assert_called_once()
        config_args = mock_types.GenerateContentConfig.call_args[1]
        self.assertEqual(config_args['system_instruction'], "System instructions")
        self.assertEqual(config_args['temperature'], 0.5)
        self.assertEqual(config_args['max_output_tokens'], 100)


class TestRetryLogic(unittest.TestCase):
    """Test the retry logic in GeminiLLMProvider."""
    
    @patch('google.genai.Client')
    @patch('google.genai.types')
    def test_retry_on_resource_exhausted(self, mock_types, mock_client):
        """Test retry logic when API returns ResourceExhausted error."""
        # Import the exception directly in the test
        from google.api_core.exceptions import ResourceExhausted

        # Set up a side_effect sequence - fail twice then succeed
        mock_response = MagicMock()
        mock_response.text = "Success after retry"
        mock_generate = mock_client.return_value.models.generate_content
        mock_generate.side_effect = [
            ResourceExhausted("Rate limit exceeded"),
            ResourceExhausted("Rate limit exceeded"),
            mock_response
        ]
        
        # Use the tenacity implementation
        provider = GeminiLLMProvider(
            api_key="fake_key",
            max_retries=3,
            min_wait=0,  # Set to 0 to make the test run faster
            max_wait=0.1
        )
        
        # This should retry and eventually succeed
        result = provider.generate(prompt="Test prompt")
        
        # Verify API was called 3 times (2 failures + 1 success)
        self.assertEqual(mock_generate.call_count, 3)
        
        # Verify we got the successful result
        self.assertEqual(result, "Success after retry")
    
    @patch('google.genai.Client')
    @patch('google.genai.types')
    def test_no_retry_on_other_exceptions(self, mock_types, mock_client):
        """Test that other exceptions don't trigger retries."""
        
        mock_response = MagicMock()
        mock_response.text = "Should not trigger retry"
        
        # Mock the API to fail with a different exception
        mock_client.return_value.models.generate_content.side_effect = [
            ValueError("API Error"), 
            mock_response
        ]
        
        provider = GeminiLLMProvider(
            api_key="fake_key",
            max_retries=3,
            min_wait=0,  # Set to 0 to make the test run faster
            max_wait=0.1
        )
        
        # Should immediately raise the exception without retrying
        with self.assertRaises(ValueError):
            provider.generate(prompt="Test prompt")
        
        # Should have called API only once (no retries)
        self.assertEqual(mock_client.return_value.models.generate_content.call_count, 1)


class TestTokenLimits(unittest.TestCase):
    """Test token limit functionality."""
    
    def setUp(self):
        """Set up a minimal provider for testing token limits."""
        class TestProvider(LLMProvider):
            def generate_completion(self, 
                                prompt: str,
                                system_prompt: Optional[str] = None,
                                temperature: float = 0.7,
                                max_tokens: Optional[int] = None,
                                **kwargs) -> str:
                # Simple implementation that just returns test response
                return "Test response"
                
        self.provider = TestProvider(
            model_name="test-model",
            input_token_limit=100,
            output_token_limit=50,
        )
    
    def test_token_limit_checks(self):
        """Test that token limit checks work correctly."""
        # Should start with no tokens
        usage = self.provider.get_token_usage()
        self.assertEqual(usage["input_tokens"], 0)
        self.assertEqual(usage["output_tokens"], 0)
        
        # Update within limits should work
        self.provider.update_token_usage(50, 25)
        self.assertTrue(self.provider.check_token_limits(prompt="Test prompt", max_tokens=10))
        
        # Verify token counts
        usage = self.provider.get_token_usage()
        self.assertEqual(usage["input_tokens"], 50)
        self.assertEqual(usage["output_tokens"], 25)
        
    def test_input_token_limit_exceeded(self):
        """Test that exceeding input token limit raises error."""
        with self.assertRaises(ValueError) as context:
            self.provider.generate(prompt="a" * 404)  # Exceeding input token limit
        self.assertIn("Estimated input token limit exceeded", str(context.exception))
        
        # Should not be able to update token usage if limits are exceeded
        self.provider.update_token_usage(99, 0)
        with self.assertRaises(ValueError) as context:
            self.provider.generate(prompt="a" * 8)
        self.assertIn("Estimated input token limit exceeded", str(context.exception))
        
    def test_output_token_limit_exceeded(self):
        """Test that exceeding output token limit raises error."""
        with self.assertRaises(ValueError) as context:
            self.provider.generate(prompt="a" * 1, max_tokens=100)
        self.assertIn("Estimated output token limit exceeded", str(context.exception))
        
        # Should not be able to update token usage if limits are exceeded
        self.provider.update_token_usage(0, 100)
        with self.assertRaises(ValueError) as context:
            self.provider.generate(prompt="a" * 1)
        self.assertIn("Estimated output token limit exceeded", str(context.exception))
        
        