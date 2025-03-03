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
    
    @abstractmethod
    def generate(self,
                 prompt: str,
                 system_prompt: Optional[str] = None,
                 temperature: float = 0.7,
                 max_tokens: Optional[int] = None,
                 **kwargs
                ) -> str:
        """Generate text from the LLM based on the prompt.

        Args:
            prompt (str): The input prompt for the model.
            system_prompt (Optional[str], optional): System prompt for the model. Defaults to None.
            temperature (float, optional): Controls randomness in generation. Defaults to 0.7.
            max_tokens (Optional[int], optional): Maximum number of tokens to generate. Defaults to None.

        Returns:
            str: The generated text.
        """
        pass


    def generate_batch(self,
                     prompts: list[str],
                     temperature: float = 0.7,
                     max_tokens: Optional[int] = None,
                     **kwargs
                    ) -> list[str]:
        """Generate text from the LLM in batch based on a list of prompts.

        Args:
            prompts (list[str]): The input prompts for the model.
            temperature (float, optional): Controls randomness in generation. Defaults to 0.7.
            max_tokens (Optional[int], optional): Maximum number of tokens to generate. Defaults to None.

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
                 api_key: Optional[str] = None,
                 min_wait: Optional[int] = 1,
                 max_wait: Optional[int] = 60,
                 max_retries: Optional[int] = 5,
    ):
        """
        Initialize a Gemini LLM provider instance.

        Args:
            model_name (str, optional): Gemini model id. Defaults to "gemini-2.0-flash".
            api_key (Optional[str], optional): Gemini API key provided by Google. Defaults to None.
            min_wait (Optional[int], optional): Minimum wait time between retries in seconds. Defaults to 1.
            max_wait (Optional[int], optional): Maximum wait time between retries in seconds. Defaults to 60.
            max_retries (Optional[int], optional): Maximum retries for failed requests. Defaults to 5.
        """
        try:
            from google import genai
            from google.api_core import exceptions
            from google.genai import types
            
        except ImportError as err:
            raise ImportError(
                "Install 'google-genai' and 'google-api-core' packages to use Gemini LLM provider."
            ) from err
        
        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name
        self.types = types
        
        # Retry settings for handling rate limits
        self.retry_settings = {}
        if min_wait is not None and max_wait is not None:
            self.retry_settings["wait"] = tenacity.wait_exponential(min=min_wait, max=max_wait)
        if max_retries is not None:
            self.retry_settings["stop"] = tenacity.stop_after_attempt(max_retries)
        self.retry_settings["retry"] = tenacity.retry_if_exception_type(exceptions.ResourceExhausted)
        
    def generate(self,
                 prompt: str,
                 system_prompt: Optional[str] = None,
                 temperature: float = 0.7,
                 max_tokens: Optional[int] = None,
                 **kwargs):
        """
        Generate text using the Gemini LLM API. Uses tenacity for retrying on rate limits.
        """
        retry_decorator = tenacity.retry(**self.retry_settings)
        
        @retry_decorator
        def _generate_with_retry():
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
        
        return _generate_with_retry()
            