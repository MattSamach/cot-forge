"""
LLM provider base class implementation.
"""

import logging
from abc import ABC, abstractmethod
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
                rate_limit_exceptions: tuple[Exception] | None = None):
        """
        Initialize an LLM provider instance.

        Args:
            min_wait: Minimum wait time between retries in seconds. Defaults to 0.0.
            max_wait: Maximum wait time between retries in seconds. Defaults to 0.0.
            max_retries: Maximum retries for failed requests. Defaults to 0.
            rate_limit_exceptions: List of exceptions to retry on.
        """        
        # Retry settings for handling rate limits
        self.retry_settings = {}
        if min_wait is not None and max_wait is not None:
            self.retry_settings["wait"] = tenacity.wait_exponential(min=min_wait, max=max_wait)
        if max_retries is not None:
            self.retry_settings["stop"] = tenacity.stop_after_attempt(max_retries)
        self.retry_settings["retry"] = tenacity.retry_if_exception_type(
            rate_limit_exceptions or [tenacity.RetryError])
    
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
        Uses the generate_completion method of the subclass LLM provider."""
        
        @tenacity.retry(**self.retry_settings)
        def _generate_with_retry():
            return self.generate_completion(prompt, system_prompt, temperature, max_tokens, **kwargs)
        return _generate_with_retry()
    
    
    def generate_batch(self,
                     prompts: list[str],
                     temperature: float = 0.7,
                     max_tokens: int | None = None,
                     **kwargs
                    ) -> list[str]:
        """Generate text from the LLM in batch based on a list of prompts.

        Args:
            prompts: The input prompts for the model.
            temperature: Controls randomness in generation. Defaults to 0.7.
            max_tokens: Maximum number of tokens to generate. Defaults to None.

        Returns:
            NotImplementedError: Batch generation is planned for future implementation.
        """
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
        config_data.update(kwargs)
            
        response = self.client.models.generate_content(
            model=self.model_name,
            config=self.types.GenerateContentConfig(
                **config_data
            ),
            contents=[prompt]
        )

        return response.text            