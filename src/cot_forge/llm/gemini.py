from typing import Optional

from .llm_provider import LLMProvider


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
                exceptions.ResourceExhausted
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