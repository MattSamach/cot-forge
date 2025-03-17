"""
This module defines an abstract base class for LLM providers and a concrete implementation for the Gemini LLM.
The `LLMProvider` class provides a common interface for interacting with different LLMs, 
including methods for generating text, managing token usage, and handling rate limits. 
It uses the `tenacity` library for retrying failed requests.
"""

import logging
from abc import ABC, abstractmethod
from threading import RLock
from typing import Optional

import tenacity

logger = logging.getLogger(__name__)

class LLMProvider(ABC):
    """
    Abstract base class for LLM providers.
    """
    
    def __init__(self, 
                min_wait: float = 0.0,
                max_wait: float = 0.0,
                max_retries: int = 0,
                rate_limit_exceptions: tuple[Exception] | None = None,
                input_token_limit: int | None = None,
                output_token_limit: int | None = None,
                ):
        """
        Initialize an LLM provider instance.

        Args:
            min_wait: Minimum wait time between retries in seconds.
            max_wait: Maximum wait time between retries in seconds.
            max_retries: Maximum retries for failed requests.
            rate_limit_exceptions: List of exceptions to retry on.
            input_token_limit: Maximum number of input tokens, for cost control.
            output_token_limit: Maximum number of output tokens, for cost control.
        """        
        # Retry settings for handling rate limits
        self.retry_settings = {}
        if min_wait is not None and max_wait is not None:
            self.retry_settings["wait"] = tenacity.wait_exponential(min=min_wait, max=max_wait)
        if max_retries is not None:
            self.retry_settings["stop"] = tenacity.stop_after_attempt(max_retries)
        self.retry_settings["retry"] = tenacity.retry_if_exception_type(
            rate_limit_exceptions or (tenacity.RetryError,))
        self.input_token_limit = input_token_limit
        self.output_token_limit = output_token_limit
        self.input_tokens = 0
        self.output_tokens = 0
        self._lock = RLock()

    def __str__(self):
        """String representation of the LLM provider."""
        return f"""{self.__class__.__name__}
        (input_tokens: {self.input_tokens}, output_tokens: {self.output_tokens})"""
        
    def __repr__(self):
        """String representation of the LLM provider."""
        return f"""{self.__class__.__name__}
        (input_tokens: {self.input_tokens}, output_tokens: {self.output_tokens})"""
        
    def get_token_usage(self):
        """Get the token usage information."""
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
        }
    
    def estimate_input_tokens(self, prompt: str) -> int:
        """Estimate the number of input tokens for a given prompt."""
        return len(prompt) / 4 

    def check_token_limits(self, prompt = None, system_prompt = None, max_tokens = None) -> bool:
        """Check if the token limits are exceeded."""
        # Estimate input token usage
        input_tokens = int(self.estimate_input_tokens(prompt))
        
        # Estimate max allowable output size
        if not max_tokens and self.output_token_limit is not None:
            # Calculate max tokens based on output token limit
            max_tokens = self.output_token_limit - self.output_tokens
        
        if self.input_token_limit is not None and self.input_tokens + input_tokens > self.input_token_limit:
            raise ValueError("Estimated input token limit exceeded")
        elif self.output_token_limit is not None and \
            (max_tokens <= 0 or self.output_tokens + max_tokens > self.output_token_limit):
            raise ValueError("Estimated output token limit exceeded")
            
        
        return True
    
    @abstractmethod
    def generate_completion(self,
                            prompt: str,
                            system_prompt: str | None = None,
                            temperature: float = 0.7,
                            max_tokens: int | None = None,
                            **kwargs
                            ) -> str:
        """Generate text from the LLM based on the prompt.

        Args:
            prompt: The input prompt for the model.
            system_prompt: System prompt for the model. Defaults to None.
            temperature: Controls randomness in generation. Defaults to 0.7.
            max_tokens: Maximum number of tokens to generate. Defaults to None.

        Returns:
            str: The generated text.
        """
        pass
    
    def generate(self,
                 prompt: str,
                 system_prompt: str | None = None,
                 temperature: float = 0.7,
                 max_tokens: int | None = None,
                 **kwargs):
        """Tenacity attempts multiple retries of generate text when 
        blocked by rate limit / resource overuse exceptions.
        Uses the generate_completion method of the subclass LLM provider.
        Also checks token limits before generating text.
        """
        
        @tenacity.retry(**self.retry_settings)
        def _generate_with_retry():
            self.check_token_limits(prompt, system_prompt, max_tokens)
            return self.generate_completion(prompt, system_prompt, temperature, max_tokens, **kwargs)
        return _generate_with_retry()
    
    def update_token_usage(self,
                           input_tokens: int,
                           output_tokens: int
                           ) -> None:
        """Thread-safe token usage update."""
        with self._lock:
            if input_tokens is not None:
                self.input_tokens += input_tokens
            if output_tokens is not None:
                self.output_tokens += output_tokens
            logger.debug(f"Token usage updated: {self.input_tokens} input, {self.output_tokens} output") 
               
    def generate_batch(self,
                       prompts: list[str],
                       temperature: float = 0.7,
                       max_tokens: int | None = None,
                       **kwargs
                       ) -> list[str]:
        """Implementation would use the provider's native batch API if available"""

        raise NotImplementedError("Batch generation is planned for future implementation.")
    
class GeminiLLMProvider(LLMProvider):
    """
    Gemini LLM provider implementation.
    """
    
    def __init__(self,
                 model_name:str = "gemini-2.0-flash",
                 api_key: str | None = None,
                 min_wait: float | None = None,
                 max_wait: float | None = None,
                 max_retries: int | None = None,
                 ):
        """
        Initialize a Gemini LLM provider instance.

        Args:
            model_name: Gemini model id. Defaults to "gemini-2.0-flash".
            api_key: API key for Gemini API. Defaults to None.
            min_wait: Minimum wait time between retries in seconds. 
                Defaults to parent class default.
            max_wait: Maximum wait time between retries in seconds. 
                Defaults to parent class default.
            max_retries: Maximum retries for failed requests. 
                Defaults to parent class default.
        """
        
        try:
            from google import genai
            from google.api_core import exceptions
            from google.genai import types
            
            rate_limit_exceptions = (
                exceptions.TooManyRequests,
                exceptions.TooManyRequests
            )
            
        except ImportError as err:
            raise ImportError(
                "Install 'google-genai' and 'google-api-core' packages to use Gemini LLM provider."
            ) from err
            
        super().__init__(min_wait=min_wait,
                         max_wait=max_wait, 
                         max_retries=max_retries,
                         rate_limit_exceptions=rate_limit_exceptions
                         )
        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name
        self.types = types
        
    def generate_completion(self,
                            prompt: str,
                            system_prompt: Optional[str] = None,
                            temperature: float = 0.7,
                            max_tokens: Optional[int] = None,
                            **kwargs):
        """
        Generate text using the Gemini LLM API.
        """        
        config_data = {"system_instruction": system_prompt} if system_prompt else {}
        config_data["temperature"] = temperature
        config_data["max_output_tokens"] = max_tokens
        llm_kwargs = kwargs.get("llm_kwargs", {})
        config_data.update(llm_kwargs)
        
        # Generate content using the Gemini API    
        response = self.client.models.generate_content(
            model=self.model_name,
            config=self.types.GenerateContentConfig(
                **config_data
            ),
            contents=[prompt]
        )
        
        # Update token usage, check limits and raise error if exceeded
        usage_metadata = response.usage_metadata
        self.update_token_usage(
            input_tokens=usage_metadata.prompt_token_count,
            output_tokens=usage_metadata.candidates_token_count
        )

        return response.text            