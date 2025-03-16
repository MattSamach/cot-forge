
import logging
from typing import Any, Callable, TypeVar

from cot_forge.reasoning.types import ReasoningNode, SearchResult

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