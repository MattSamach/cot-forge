
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