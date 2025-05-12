import logging
import os

from .llm_provider import LLMProvider

logger = logging.getLogger(__name__)


class OpenAIProvider(LLMProvider):
  """
  OpenAI LLM provider implementation.
  """

  def __init__(
      self,
      model_name: str = "gpt-4o",
      api_key: str | None = None,
      input_token_limit: int | None = None,
      output_token_limit: int | None = None,
      min_wait: float | None = None,
      max_wait: float | None = None,
      max_retries: int | None = None,
  ):
    """
    Initialize an OpenAI LLM provider instance.

    Args:
        model_name (str): OpenAI model ID. Defaults to "gpt-4o".
        api_key (str | None): API key for the OpenAI API. Required to authenticate requests.
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

    try:
      from tiktoken import encoding_for_model
      self.enc = encoding_for_model(model_name)
    except ImportError:
      logging.warning(
          "'tiktoken' package not found. Token counting will not be accurate.")
      self.enc = None

    super().__init__(
        model_name=model_name,
        min_wait=min_wait,
        max_wait=max_wait,
        max_retries=max_retries,
        rate_limit_exceptions=rate_limit_exceptions,
        input_token_limit=input_token_limit,
        output_token_limit=output_token_limit,
    )
    # Set the API key for OpenAI
    if api_key is None:
      api_key = os.getenv("OPENAI_API_KEY")
    if api_key is None:
      raise ValueError("API key for OpenAI not provided.")
    self.client = OpenAI(api_key=api_key)
    self.model_name = model_name

  def generate_completion(self,
                          prompt: str,
                          system_prompt: str | None = None,
                          temperature: float = 0.7,
                          max_tokens: int | None = None,
                          **kwargs):
    """
    Generate text using the OpenAI LLM API.

    This method sends a prompt to the OpenAI API and retrieves the generated text.
    It also updates token usage statistics and enforces token limits.

    Args:
        prompt (str): The input prompt for the model.
        system_prompt (Optional[str]): An optional system instruction to guide the model's behavior.
        temperature (float): Controls randomness in generation. Higher values produce more random outputs.
            Defaults to 0.7.
        max_tokens (Optional[int]): The maximum number of output tokens to generate. Defaults to None.
        **kwargs: Additional arguments for the OpenAI API. For example:
            - `llm_kwargs` (dict): A dictionary of additional configuration options for the API.

    Returns:
        str: The generated text from the OpenAI API.

    Raises:
        ValueError: If token limits are exceeded.
        RateLimitError: If the API rate limit is exceeded.
        Exception: For other API-related errors.

    Example:
        ```python
        provider = OpenAILLMProvider(api_key="your_api_key")
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
        "max_output_tokens": max_tokens
    }
    llm_kwargs = kwargs.get("llm_kwargs", {})
    config_data.update(llm_kwargs)

    # Generate messages for the API
    messages = [] if not system_prompt else [
        {"role": "developer", "content": system_prompt}]
    messages.append({"role": "user", "content": prompt})

    # Generate content using the OpenAI API
    response = self.client.responses.create(
        model=self.model_name,
        input=messages,
        **config_data
    )

    # Update token usage, check limits and raise error if exceeded
    usage_metadata = response.usage
    self.update_token_usage(
        input_tokens=usage_metadata.input_tokens,
        output_tokens=usage_metadata.output_tokens
    )

    return response.output_text

  def estimate_input_tokens(self, prompt: str) -> int:
    """
    Estimate the number of input tokens for a given prompt.
    If the 'tiktoken' package is not available, a rough estimate is returned.
    """
    if not self.enc:
      # If 'tiktoken' is not available, return a rough estimate
      return len(prompt) // 4
    # Use 'tiktoken' to count tokens accurately
    try:
      return len(self.enc.encode(prompt))
    except Exception:
      # logger.warning(f"Error estimating input tokens: {e}")
      return len(prompt) // 4
