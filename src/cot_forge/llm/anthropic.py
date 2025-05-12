import logging
import os

from .llm_provider import LLMProvider

logger = logging.getLogger(__name__)


class AnthropicProvider(LLMProvider):
  """
  Anthropic LLM provider implementation.
  """

  def __init__(
      self,
      model_name: str = "claude-3-5-sonnet-20241022",
      api_key: str | None = None,
      input_token_limit: int | None = None,
      output_token_limit: int | None = None,
      min_wait: float | None = None,
      max_wait: float | None = None,
      max_retries: int | None = None,
  ):
    """
    Initialize an Anthropic LLM provider instance.

    Args:
        model_name (str): Anthropic model ID. Defaults to "claude-3-7-sonnet-20250219".
        api_key (str | None): API key for the Anthropic API. Required to authenticate requests.
        input_token_limit (int | None): Maximum number of input tokens, for cost control.
        output_token_limit (int | None): Maximum number of output tokens, for cost control.
        min_wait (float | None): Minimum wait time between retries in seconds. 
            Defaults to the parent class's behavior.
        max_wait (float | None): Maximum wait time between retries in seconds. 
            Defaults to the parent class's behavior.
        max_retries (int | None): Maximum number of retries for failed requests. 
            Defaults to the parent class's behavior.

    Raises:
        ImportError: If the 'anthropic' package is not installed.
    """

    try:
      from anthropic import Anthropic, RateLimitError

      rate_limit_exceptions = (
          RateLimitError
      )

    except ImportError as err:
      raise ImportError(
          "Install 'anthropic' package to use Anthropic LLM provider."
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
    # Initialize the Anthropic client
    # If api_key is None, it will be read from the environment variable ANTHROPIC_API_KEY
    if api_key is None:
      api_key = os.getenv("ANTHROPIC_API_KEY")
    if api_key is None:
      raise ValueError("API key is required for Anthropic LLM provider.")
    self.client = Anthropic(api_key=api_key)
    self.model_name = model_name

  def generate_completion(self,
                          prompt: str,
                          system_prompt: str | None = None,
                          temperature: float = 0.7,
                          max_tokens: int | None = 1024,
                          **kwargs):
    """
    Generate text using the Anthropic LLM API.

    This method sends a prompt to the Anthropic API and retrieves the generated text.
    It also updates token usage statistics and enforces token limits.

    Args:
        prompt (str): The input prompt for the model.
        system_prompt (Optional[str]): An optional system instruction to guide the model's behavior.
        temperature (float): Controls randomness in generation. Higher values produce more random outputs.
            Defaults to 0.7.
        max_tokens (Optional[int]): The maximum number of output tokens to generate. Defaults to None.
        **kwargs: Additional arguments for the Anthropic API. For example:
            - `llm_kwargs` (dict): A dictionary of additional configuration options for the API.

    Returns:
        str: The generated text from the Anthropic API.

    Raises:
        ValueError: If token limits are exceeded.
        RateLimitError: If the API rate limit is exceeded.
        Exception: For other API-related errors.

    Example:
        ```python
        provider = AnthropicLLMProvider(api_key="your_api_key")
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
    }
    # Anthropic requires the max_tokens parameter to be set, so default to 1024 if not provided
    config_data["max_tokens"] = max_tokens or 1024
    if system_prompt:
      config_data["system"] = system_prompt
    llm_kwargs = kwargs.get("llm_kwargs", {})
    config_data.update(llm_kwargs)

    # Generate messages for the API
    messages = [{"role": "user", "content": prompt}]

    # Generate content using the Anthropic API
    response = self.client.messages.create(
        model=self.model_name,
        messages=messages,
        **config_data
    )

    # Update token usage, check limits and raise error if exceeded
    usage_metadata = response.usage
    self.update_token_usage(
        input_tokens=usage_metadata.input_tokens,
        output_tokens=usage_metadata.output_tokens
    )

    return response.content[0].text

  def estimate_input_tokens(self, prompt: str) -> int:
    """
    Estimate the number of input tokens for a given prompt.
    """
    try:
      response = self.client.messages.count_tokens(
          model=self.model_name,
          messages=[{"role": "user", "content": prompt}]
      )
      return response.input_tokens
    except Exception:
      # logger.warning(f"Error estimating input tokens: {e}")
      return len(prompt) // 4
