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
    
    def __init__(self,
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
        self.model_name = model_name
        
        # Retry settings for handling rate limits
        self.retry_settings = {}
        if min_wait is not None and max_wait is not None:
            self.retry_settings["wait"] = tenacity.wait_exponential(min=min_wait, max=max_wait)
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

    def __str__(self):
        """String representation of the LLM provider."""
        return ("f{self.__class__.__name__}\n",
        f"\t(model_name: {self.model_name})\n",
        f"\t(input_tokens: {self.input_tokens}, output_tokens: {self.output_tokens})")
        
    def __repr__(self):
        """String representation of the LLM provider for developers."""
        return (f"{self.__class__.__name__}\n",
        f"\t(model_name: {self.model_name})\n",
        f"\t(input_tokens: {self.input_tokens}, output_tokens: {self.output_tokens})\n",
        f"\t(input_token_limit: {self.input_token_limit} ",
        f"output_token_limit: {self.output_token_limit})\n",
        f"\t(retry_settings: {self.retry_settings})\n")
        
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
            logger.debug(f"Token usage updated: {self.input_tokens} input, {self.output_tokens} output") 
               
    def generate_batch(self,
                       prompts: list[str],
                       temperature: float = 0.7,
                       max_tokens: int | None = None,
                       **kwargs
                       ) -> list[str]:
        """Implementation would use the provider's native batch API if available"""

        raise NotImplementedError("Batch generation is planned for future implementation.")