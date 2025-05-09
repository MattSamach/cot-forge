import logging
import os

from .llm_provider import LLMProvider

logger = logging.getLogger(__name__)


class OpenRouterProvider(LLMProvider):
  """
  Open Router LLM provider implementation.
  """

  def __init__(
      self,
      model_name: str = "deepseek/deepseek-chat-v3-0324:free",
      api_key: str | None = None,
      input_token_limit: int | None = None,
      output_token_limit: int | None = None,
      min_wait: float | None = None,
      max_wait: float | None = None,
      max_retries: int | None = None,
  ):
    """
    Initialize an OpenRouter LLM provider instance.

    Args:
        model_name (str): OpenRouter model ID. Defaults to "deepseek/deepseek-chat-v3-0324:free".
        api_key (str | None): API key for the OpenRouter API. Required to authenticate requests.
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
          "Install 'openai' package to use OpenRouter LLM provider."
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
    # Check if the API key is provided
    if api_key is None:
      api_key = os.getenv("OPENROUTER_API_KEY")
    if api_key is None:
      raise ValueError(
          "API key is required. "
          "Set it as an argument or in the environment variable 'OPENROUTER_API_KEY'."
      )
    self.client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )
    self.model_name = model_name

  def generate_completion(self,
                          prompt: str,
                          system_prompt: str | None = None,
                          temperature: float = 0.7,
                          max_tokens: int | None = 1024,
                          **kwargs):
    """
    Generate text using the Open Router LLM API.

    This method sends a prompt to the Open Router API and retrieves the generated text.
    It also updates token usage statistics and enforces token limits.

    Args:
        prompt (str): The input prompt for the model.
        system_prompt (Optional[str]): Not used in this implementation.
            Defaults to None.
        temperature (float): Controls randomness in generation. Higher values produce more random outputs.
            Defaults to 0.7.
        max_tokens (Optional[int]): The maximum number of output tokens to generate. Defaults to None.
        **kwargs: Additional arguments for the Open Router API. For example:
            - `llm_kwargs` (dict): A dictionary of additional configuration options for the API.

    Returns:
        str: The generated text from the Open Router API.

    Raises:
        ValueError: If token limits are exceeded.
        RateLimitError: If the API rate limit is exceeded.
        Exception: For other API-related errors.

    Example:
        ```python
        provider = OpenRouterProvider(api_key="your_api_key")
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

    # Generate content using the Open Router API
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
