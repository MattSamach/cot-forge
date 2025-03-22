
import logging
import time
from typing import Any, Literal

from cot_forge.llm import LLMProvider
from cot_forge.utils.parsing import extract_cot

logger = logging.getLogger(__name__)
    
def execute_with_fallback(
        operation_name: str,
        operation_func: callable,
        args: tuple = (),
        kwargs: dict = None,
        on_error: Literal["continue", "raise", "retry"] = "continue",
        max_retries: int = 3,
        retry_delay: float = 1.0,
        fallback_value: Any = None,
        logger: logging.Logger = logger
    ) -> tuple[Any, str | None]:
        """
        This function provides a robust wrapper for executing operations that might fail,
        with configurable error handling strategies: continuing with a fallback value,
        raising the error, or retrying the operation.
        Parameters:
            operation_name (str): Descriptive name of the operation for logging purposes
            operation_func (callable): The function to execute
            args (tuple, optional): Positional arguments to pass to operation_func
            kwargs (dict, optional): Keyword arguments to pass to operation_func
            on_error (Literal["continue", "raise", "retry"], optional): Error handling strategy:
                - "continue": Return fallback value and error message
                - "raise": Raise a RuntimeError with details
                - "retry": Attempt to retry the operation up to max_retries
            max_retries (int, optional): Maximum number of retry attempts if on_error="retry"
            retry_delay (float, optional): Delay in seconds between retry attempts
            fallback_value (Any, optional): Value to return if operation fails and on_error="continue"
            logger (logging.Logger, optional): Logger instance for recording events
        Returns:
            tuple[Any, str | None]: A tuple containing:
                - The operation result on success, or fallback_value on failure
                - None on success, or error message string on failure
        Raises:
            RuntimeError: If the operation fails and on_error="raise"
        Example:
            >>> result, error = execute_with_fallback(
            ...     "fetch_data",
            ...     api.get_data,
            ...     kwargs={"endpoint": "/users"},
            ...     on_error="retry",
            ...     fallback_value=[]
            ... )
            >>> if error:
            ...     print(f"Warning: {error}")
            >>> process_data(result)
        """
        
        kwargs = kwargs or {}
        attempts = 0
        last_error = None
        
        while True:
            attempts += 1
            try:
                result = operation_func(*args, **kwargs)
                return result, None  # Success case - return result with no error
            except Exception as e:
                error_msg = f"{operation_name} failed: {str(e)}"
                last_error = e
                
                if on_error == "retry" and attempts <= max_retries:
                    logger.warning(f"{error_msg}. Retrying ({attempts}/{max_retries})...")
                    time.sleep(retry_delay)
                    continue
                    
                break
        
        # If we exhausted retries but still have errors
        if last_error and on_error == "retry" and attempts > 1:
            logger.error(f"All {max_retries} retry attempts failed for {operation_name}")
            
        # Handle final error state based on error action
        if last_error:
            error_msg = f"{operation_name} failed: {str(last_error)}"
            
            if on_error == "raise":
                logger.error(error_msg)
                raise RuntimeError(error_msg)
            elif on_error == "continue":
                logger.warning(f"{error_msg}. Continuing with fallback value.")
                return fallback_value, error_msg
        
        # This should never be reached in practice
        logger.error(f"Unexpected code path in execute_with_fallback for {operation_name}")
        return None, f"Unexpected error in {operation_name}"
    
def generate_and_parse_cot(
        search_llm: LLMProvider,
        prompt: str,
        llm_kwargs: dict[str, Any] = None,
        on_error: Literal["continue", "raise", "retry"] = "retry",
        max_retries: int = 3,
        retry_delay: float = 1.0,
        logger: logging.Logger = logger,
    ) -> tuple[str, list[dict[str, str]]]:
        """
        Generate and parse the chain of thought (CoT) from the LLM.
        
        Args:
            search_llm: The LLM provider to use for generation
            prompt: The prompt to send to the LLM
            llm_kwargs: Additional kwargs for LLM generation
            on_error: How to handle errors during generation
            max_retries: Maximum number of retries if on_error="retry"
            retry_delay: Delay between retries in seconds
        Returns:
            tuple[str, list[dict[str, str]]]: The generated response and parsed CoT
        Raises:
            RuntimeError: If the operation fails and on_error="raise" or "retry"
        Example:
            >>> response, cot = generate_and_parse_cot(
            ...     search_llm,
            ...     prompt="What is the capital of France?",
            ...     llm_kwargs={"temperature": 0.7},
            ...     on_error="retry",
            ...     max_retries=5,
            ...     retry_delay=2.0
            ... )
            >>> print("Response:", response)
            >>> print("CoT:", cot)
        """
        
        def helper_function():
            # Generate the response using the LLM
            response = search_llm.generate(prompt, **(llm_kwargs or {}))
            # Extract the CoT from the response
            cot = extract_cot(response)
            return response, cot
        
        # Execute the operation with error handling
        result, error_msg = execute_with_fallback(
            operation_name="LLM generation and CoT extraction",
            operation_func=helper_function,
            on_error=on_error,
            max_retries=max_retries,
            retry_delay=retry_delay,
            logger=logger,
            fallback_value=(None, None)
        )
        if error_msg and (on_error == "raise" or on_error == "retry"):
            # Log the error and raise an exception
            logger.error(f"LLM generation and CoT extraction failed: {error_msg}")
            raise RuntimeError(f"LLM generation and CoT extraction failed: {error_msg}")
        elif error_msg and on_error == "continue":
            # Log the error but continue
            logger.error(f"LLM generation and CoT extraction failed: {error_msg}")
            return None, None
        
        # If the operation was successful, unpack the result
        response, cot = result
        if response is None or cot is None:
            # Handle the case where the operation failed
            logger.error("LLM generation and CoT extraction returned None")
            return None, None
        # Return the generated response and parsed CoT
        return response, cot