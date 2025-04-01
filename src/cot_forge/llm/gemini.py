import os

from .llm_provider import LLMProvider


class GeminiProvider(LLMProvider):
    """
    Gemini LLM provider implementation.
    """
    
    def __init__(self,
                 model_name:str = "gemini-2.0-flash",
                 api_key: str | None = None,
                 input_token_limit: int | None = None,
                 output_token_limit: int | None = None,
                 min_wait: float | None = None,
                 max_wait: float | None = None,
                 max_retries: int | None = None,
                 ):
        """
        Initialize a Gemini LLM provider instance.

        Args:
            model_name (str): Gemini model ID. Defaults to "gemini-2.0-flash".
            api_key (str | None): API key for the Gemini API. Required to authenticate requests.
            input_token_limit (int | None): Maximum number of input tokens, for cost control.
            output_token_limit (int | None): Maximum number of output tokens, for cost control.
            min_wait (float | None): Minimum wait time between retries in seconds. 
                Defaults to the parent class's behavior.
            max_wait (float | None): Maximum wait time between retries in seconds. 
                Defaults to the parent class's behavior.
            max_retries (int | None): Maximum number of retries for failed requests. 
                Defaults to the parent class's behavior.

        Raises:
            ImportError: If the required `google-genai` or `google-api-core` packages are not installed.
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
            
        super().__init__(
            model_name=model_name,
            min_wait=min_wait,
            max_wait=max_wait, 
            max_retries=max_retries,
            rate_limit_exceptions=rate_limit_exceptions,
            input_token_limit=input_token_limit,
            output_token_limit=output_token_limit,
        )
        # Set up the Gemini client
        # If api_key is not provided, try to read from the environment variable GOOGLE_API_KEY
        if not api_key:
            api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("API key is required for Gemini LLM provider.")
        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name
        self.types = types
        
    def generate_completion(self,
                            prompt: str,
                            system_prompt: str | None = None,
                            temperature: float = 0.7,
                            max_tokens: int | None = None,
                            **kwargs):
        """
        Generate text using the Gemini LLM API.

        This method sends a prompt to the Gemini API and retrieves the generated text.
        It also updates token usage statistics and enforces token limits.

        Args:
            prompt (str): The input prompt for the model.
            system_prompt (Optional[str]): An optional system instruction to guide the model's behavior.
            temperature (float): Controls randomness in generation. Higher values produce more random outputs.
                Defaults to 0.7.
            max_tokens (Optional[int]): The maximum number of tokens to generate. Defaults to None.
            **kwargs: Additional arguments for the Gemini API. For example:
                - `llm_kwargs` (dict): A dictionary of additional configuration options for the API.

        Returns:
            str: The generated text from the Gemini API.

        Raises:
            ValueError: If token limits are exceeded.
            google.api_core.exceptions.GoogleAPIError: If the Gemini API request fails.

        Example:
            ```python
            provider = GeminiProvider(api_key="your_api_key")
            response = provider.generate_completion(
                prompt="Write a poem about the ocean.",
                temperature=0.8,
                max_tokens=100
            )
            print(response)
            ```
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