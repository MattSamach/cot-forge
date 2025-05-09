"""
Abstract base class for LLM providers, defining a common interface for LLM interactions,
including text generation, token management, and rate limit handling using `tenacity` for retries.
"""

import logging
from abc import ABC, abstractmethod
from threading import RLock

import tenacity

logger = logging.getLogger(__name__)


class LLMProvider(ABC):
  """
  Abstract base class for LLM providers.
  """

  def __init__(
      self,
      model_name: str,
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
        model_name: The name of the model.
        min_wait: Minimum wait time between retries in seconds.
        max_wait: Maximum wait time between retries in seconds.
        max_retries: Maximum retries for failed requests.
        rate_limit_exceptions: List of exceptions to retry on.
        input_token_limit: Maximum number of input tokens, for cost control.
        output_token_limit: Maximum number of output tokens, for cost control.
    """
    # Model name
    self.model_name = model_name
    # Rate limit settings
    self.min_wait = min_wait
    self.max_wait = max_wait
    self.max_retries = max_retries
    self.rate_limit_exceptions = rate_limit_exceptions or (
        tenacity.RetryError,)

    # Retry settings for handling rate limits
    self.retry_settings = {}
    if min_wait is not None and max_wait is not None:
      self.retry_settings["wait"] = tenacity.wait_exponential(
          min=min_wait, max=max_wait)
    if max_retries is not None:
      self.retry_settings["stop"] = tenacity.stop_after_attempt(max_retries)
    self.retry_settings["retry"] = tenacity.retry_if_exception_type(
        rate_limit_exceptions or (tenacity.RetryError,))

    # Token attributes
    self.input_token_limit = input_token_limit
    self.output_token_limit = output_token_limit
    self.input_tokens = 0
    self.output_tokens = 0

    # Mutual exclusion lock for thread-safe token updates
    self._lock = RLock()

  def get_token_usage(self) -> dict:
    """
    Retrieve the current token usage statistics.

    Returns:
        dict: A dictionary with the following keys:
            - "input_tokens" (int): The total number of input tokens used.
            - "output_tokens" (int): The total number of output tokens used.
    """
    return {
        "input_tokens": self.input_tokens,
        "output_tokens": self.output_tokens,
    }

  def estimate_input_tokens(self, prompt: str) -> int:
    """Estimate the number of input tokens for a given prompt."""
    return len(prompt) / 4

  def check_token_limits(self, prompt=None, system_prompt=None, max_tokens=None) -> bool:
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
    """
    Generate text with retries using the LLM provider.

    This method uses Tenacity to retry text generation in case of rate limit
    or resource overuse exceptions. It checks token limits before calling
    the `generate_completion` method of the subclass.

    Args:
        prompt (str): The input prompt for the model.
        system_prompt (str | None): Optional system prompt for the model.
        temperature (float): Controls randomness in generation. Defaults to 0.7.
        max_tokens (int | None): Maximum number of tokens to generate. Default None.
        **kwargs: Additional arguments for the LLM provider.

    Returns:
        str: The generated text.
    """

    @tenacity.retry(**self.retry_settings)
    def _generate_with_retry():
      self.check_token_limits(prompt, system_prompt, max_tokens)
      return self.generate_completion(prompt, system_prompt, temperature, max_tokens, **kwargs)
    return _generate_with_retry()

  def update_token_usage(self,
                         input_tokens: int,
                         output_tokens: int):
    """
    Update the token usage counters in a thread-safe manner.

    Args:
        input_tokens (int): The number of input tokens to add. Can be None.
        output_tokens (int): The number of output tokens to add. Can be None.
    """
    with self._lock:
      if input_tokens is not None:
        self.input_tokens += input_tokens
      if output_tokens is not None:
        self.output_tokens += output_tokens
      # logger.debug(
      #     f"Token usage updated: {self.input_tokens} input, {self.output_tokens} output"
      # )

  def __str__(self):
    """String representation of the LLM provider."""
    return (f"{self.__class__.__name__} "
            "(model: {self.model_name}, tokens: {self.input_tokens}/{self.output_tokens})")

  def __repr__(self):
    """String representation of the LLM provider for developers."""
    return (f"{self.__class__.__name__}(model_name='{self.model_name}', "
            f"input_token_limit={self.input_token_limit}, output_token_limit={self.output_token_limit}, "
            f"input_tokens={self.input_tokens}, output_tokens={self.output_tokens})")

  def to_dict(self) -> dict:
    """Convert the LLM provider to a dictionary representation."""
    return {
        "model_name": self.model_name,
        "input_token_limit": self.input_token_limit,
        "output_token_limit": self.output_token_limit,
    }

  @classmethod
  def from_dict(
      cls,
      data: dict,
      with_rate_limit: bool = True,
  ) -> "LLMProvider":
    """
    Create an LLM provider instance from a dictionary representation.
    Args:
        cls: The class on which this method is called.
        data (dict): Dictionary containing LLMProvider configuration.
        with_token_usage (bool, optional): Whether to include token usage data from input dictionary.
            If False, token counters will be initialized to 0. Defaults to False.
        with_rate_limit (bool, optional): Whether to include rate limit parameters from input dictionary.
            If False, token limits will be set to None. Defaults to True.
    Returns:
        LLMProvider: A new instance of the LLMProvider class initialized with values from the dictionary.
    The dictionary can contain the following keys:
        - model_name: Name of the language model
        - input_token_limit (optional): Maximum allowed input tokens
        - output_token_limit (optional): Maximum allowed output tokens
    """

    input_token_limit = data.get(
        "input_token_limit") if with_rate_limit else None
    output_token_limit = data.get(
        "output_token_limit") if with_rate_limit else None

    # Create an instance of the class
    return cls(
        # Values that can be reset
        input_token_limit=input_token_limit,
        output_token_limit=output_token_limit,

        # Values to retrieve from the data
        model_name=data.get("model_name"),
    )
