import logging
from typing import Optional

from .llm_provider import LLMProvider

logger = logging.getLogger(__name__)

class HuggingFaceProvider(LLMProvider):
    """
    Hugging Face LLM provider implementation.
    """
    
    def __init__(
        self,
        model_name:str = "google/gemma-2-2b-it",
        api_key: str | None = None,
        input_token_limit: int | None = None,
        output_token_limit: int | None = None,
        min_wait: float | None = None,
        max_wait: float | None = None,
        max_retries: int | None = None,
    ):
        """
        Initialize an Hugging Face LLM provider instance.

        Args:
            model_name (str): Hugging Face model ID. Defaults to "google/gemma-2-2b-it".
            api_key (str | None): API key for the Hugging Face API. Required to authenticate requests.
                Also called "access token" in Hugging Face documentation.
            input_token_limit (int | None): Maximum number of input tokens, for cost control.
            output_token_limit (int | None): Maximum number of output tokens, for cost control.
            min_wait (float | None): Minimum wait time between retries in seconds. 
                Defaults to the parent class's behavior.
            max_wait (float | None): Maximum wait time between retries in seconds. 
                Defaults to the parent class's behavior.
            max_retries (int | None): Maximum number of retries for failed requests. 
                Defaults to the parent class's behavior.

        Raises:
            ImportError: If the 'openai' package is not installed.
        """
        
        try:
            from openai import OpenAI, RateLimitError
            
            rate_limit_exceptions = (
                RateLimitError
            )
            
        except ImportError as err:
            raise ImportError(
                "Install 'openai' package to use OpenAI LLM provider."
            ) from err

        
        super().__init__(
            model_name=model_name,
            min_wait=min_wait,
            max_wait=max_wait, 
            max_retries=max_retries,
            rate_limit_exceptions=rate_limit_exceptions,
            input_token_limit=input_token_limit,
            output_token_limit=output_token_limit,
        )
        self.client = OpenAI(
            base_url="https://router.huggingface.co/hf-inference/v1",
            api_key=api_key
        )
        self.model_name = model_name
        
    def generate_completion(self,
                            prompt: str,
                            system_prompt: Optional[str] = None,
                            temperature: float = 0.7,
                            max_tokens: Optional[int] = 1024,
                            **kwargs):
        """
        Generate text using the Hugging Face LLM API.

        This method sends a prompt to the Hugging Face API and retrieves the generated text.
        It also updates token usage statistics and enforces token limits.

        Args:
            prompt (str): The input prompt for the model.
            system_prompt (Optional[str]): Not used in this implementation.
                Defaults to None.
            temperature (float): Controls randomness in generation. Higher values produce more random outputs.
                Defaults to 0.7.
            max_tokens (Optional[int]): The maximum number of output tokens to generate. Defaults to None.
            **kwargs: Additional arguments for the Hugging Face API. For example:
                - `llm_kwargs` (dict): A dictionary of additional configuration options for the API.

        Returns:
            str: The generated text from the Hugging Face API.

        Raises:
            ValueError: If token limits are exceeded.
            RateLimitError: If the API rate limit is exceeded.
            Exception: For other API-related errors.

        Example:
            ```python
            provider = HuggingFaceLLMProvider(api_key="your_api_key")
            response = provider.generate_completion(
                prompt="Write a poem about the ocean.",
                temperature=0.8,
                max_tokens=100
            )
            print(response)
            ```
        """
        config_data = {
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        llm_kwargs = kwargs.get("llm_kwargs", {})
        config_data.update(llm_kwargs)
        
        # Generate messages for the API
        messages = [{"role": "user", "content": prompt}]
        
        # Generate content using the Hugging Face API    
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            **config_data
        )
        
        # Update token usage, check limits and raise error if exceeded
        usage_metadata = response.usage
        self.update_token_usage(
            input_tokens=usage_metadata.prompt_tokens,
            output_tokens=usage_metadata.completion_tokens
        )

        return response.choices[0].message.content