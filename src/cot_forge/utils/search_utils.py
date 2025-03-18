
import logging
import time
from typing import Any, Callable, Literal, TypeVar

from cot_forge.llm import LLMProvider
from cot_forge.reasoning.types import ReasoningNode, SearchResult
from cot_forge.utils.parsing import extract_cot

T = TypeVar('T')

def create_error_handler(logger: logging.Logger) -> Callable:
    """Factory function that creates an error handler for search operations.
    
    Args:
        logger: The logger to use for logging errors.
        
    Returns:
        A function that handles errors during search operations.
    """
    def handle_error(func: Callable[..., T],
                     node: ReasoningNode = None,
                     metadata: dict[str, Any] = {},
                     **kwargs) -> T | SearchResult:
        """Handle errors during search by logging and returning a failure result.
        
        Args:
            func: The function to execute.
            node: The current reasoning node. Defaults to None.
            metadata: Additional metadata for the search result. Defaults to None.
            *args: Additional positional arguments for the function.
            **kwargs: Additional keyword arguments for the function.
            
        Returns:
            Either the result of the function or a SearchResult in case of error.
        """
        try:
            return func(**kwargs)
        except Exception as e:
            logger.error(f"Error in search: {e}")
            error_msg = f"{type(e).__name__}: {str(e)}"
            
            return {
                "final_node": None,
                "all_terminal_nodes": [None],  # Changed from [] to [None]
                "success": False,
                "final_answer": None,
                "metadata": {**metadata, "error": error_msg}
            }
    
    return handle_error

def initialize_blank_node(parent: ReasoningNode) -> ReasoningNode:
    """Initialize a blank reasoning node pointing to a given parent
    
    Args:
        parent: The parent node to point to.
        
    Returns:
        A ReasoningNode initialized with the strategy and question.
    """
    return ReasoningNode(
        strategy=None,
        prompt="",
        response="",
        cot=None,
        parent=parent
    )
    
def try_operation(
    operation_name: str,
    operation_func: callable,
    args: tuple = (),
    kwargs: dict = None,
    fallback_value = None,
    fallback_function: Callable[[Exception, dict], Any] = None,
    error_action: str = "log",  # "log", "raise", "return",
    logger: logging.Logger = logging.getLogger(__name__),
    *,
    context: dict = None
) -> tuple[Any, str | None]:
    """
    Execute an operation with standardized error handling based on context.
    
    Args:
        operation_name: Name of the operation for logging purposes
        operation_func: Function to execute
        args: Positional arguments for the function
        kwargs: Keyword arguments for the function
        fallback_value: Value to return if operation fails
        fallback_function: Function to call if operation fails
        error_action: What to do on error - log, raise, or return
        context: Additional context that might affect error handling
    
    Returns:
        tuple[result, error_msg]: Result of operation and error message if any
    """
    kwargs = kwargs or {}
    context = context or {}
    
    try:
        result = operation_func(*args, **kwargs)
        return result, None
    except Exception as e:
        error_msg = f"Error in {operation_name}: {e}"
        logger.error(error_msg)
        
        # Context-specific handling with fallback function
        if fallback_function:
            return fallback_function(e, context), error_msg
            
        # General error handling based on error_action
        if error_action == "raise":
            raise ValueError(error_msg)
        
        return fallback_value, error_msg  # For both "log" and "return"
    

def execute_with_fallback(
        operation_name: str,
        operation_func: callable,
        args: tuple = (),
        kwargs: dict = None,
        on_error: Literal["continue", "raise", "retry"] = "continue",
        max_retries: int = 3,
        retry_delay: float = 1.0,
        fallback_value: Any = None,
        logger: logging.Logger = logging.getLogger(__name__)
    ):
        """
        Execute an operation with standardized error handling including retry capability.
        
        Args:
            operation_name: Name of the operation for logging
            operation_func: The function to execute
            args: Positional arguments for the function
            kwargs: Keyword arguments for the function
            on_error: How to handle errors:
                - "continue": Return fallback value without error message
                - "raise": Return error message for caller to handle
                - "retry": Retry the operation up to max_retries times
            max_retries: Maximum number of retry attempts if on_error="retry"
            retry_delay: Seconds to wait between retry attempts
            fallback_value: Value to return on error with on_error="continue"
            logger: Logger to use for logging errors and retries
            
        Returns:
            tuple[Any, str | None]: (result of operation, error_message if any)
        """
        
        kwargs = kwargs or {}
        attempts = 0
        last_error = None
        
        # Define our custom fallback handler for try_operation
        def fallback_handler(exception, ctx):
            error_action = ctx.get("on_error", "continue")
            if error_action == "continue":
                # Return fallback but don't propagate error
                return fallback_value, None
            elif error_action == "retry":
                # Return a marker to trigger retry
                return fallback_value, f"Retry: {str(exception)}"
            else:  # "raise" or any other value
                # Return fallback with error for caller to handle
                return fallback_value, str(exception)
        
        while True:
            attempts += 1
            
            # Execute operation with appropriate error handling
            result, error_msg = try_operation(
                operation_name=operation_name,
                operation_func=operation_func,
                args=args,
                kwargs=kwargs,
                fallback_value=fallback_value,
                fallback_function=fallback_handler,
                error_action="return",
                context={"on_error": on_error},
                logger=logger,
            )
            
            # Successful operation or non-retry error handling
            if not error_msg or on_error != "retry" or attempts >= max_retries:
                break
                
            # Retry logic
            last_error = error_msg
            logger.info(f"Retry attempt {attempts}/{max_retries} for {operation_name}: {error_msg}")
            if attempts < max_retries:
                time.sleep(retry_delay)
        
        # If we exhausted retries but still have errors
        if error_msg and on_error == "retry" and attempts > 1:
            logger.warning(f"{operation_name} failed after {attempts} attempts: {error_msg}")
            
        # Handle final error state based on error action
        if error_msg:
            if on_error == "continue":
                # We continue with a failure but no error
                return result, None
            elif on_error == "retry" and attempts >= max_retries:
                # Convert to a raise after max retries
                return result, f"Max retries ({max_retries}) exceeded: {last_error}"
        
        # Return the final result and possibly error
        return result, error_msg
    
def generate_and_parse_cot(
        reasoning_llm: LLMProvider,
        prompt: str,
        llm_kwargs: dict[str, Any] = None,
        on_error: Literal["continue", "raise", "retry"] = "retry",
        max_retries: int = 3,
        retry_delay: float = 1.0,
        logger: logging.Logger = logging.getLogger(__name__),
    ) -> tuple[str, list[dict[str, str]]]:
        """
        Generate and parse the chain of thought (CoT) from the LLM.
        
        Args:
            reasoning_llm: The LLM provider to use for generation
            prompt: The prompt to send to the LLM
            llm_kwargs: Additional kwargs for LLM generation
            on_error: How to handle errors during generation
            max_retries: Maximum number of retries if on_error="retry"
            retry_delay: Delay between retries in seconds
        Returns:
            tuple[str, list[dict[str, str]]]: The generated response and parsed CoT
        """
        
        def helper_function():
            # Generate the response using the LLM
            response = reasoning_llm.generate(prompt, **(llm_kwargs or {}))
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