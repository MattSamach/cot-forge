"""
Abstract base class for verifiers in the CoT-Forge reasoning framework.

Verifiers are used to check the correctness of a reasoning node's answer
against a ground truth answer. This module defines the abstract interface
that all verifier implementations must follow.

Key features:
1. Common initialization parameters for all verifiers
2. A standard verification interface
3. Callable interface for convenient usage

Example usage:
    ```python
    from cot_forge.reasoning.verifiers import ExactMatchVerifier
    
    # Create a verifier
    verifier = ExactMatchVerifier(
        name="exact_match",
        description="Checks if answers match exactly"
    )
    
    # Verify a reasoning node
    is_correct, explanation = verifier(
        node=reasoning_node,
        question="What is 2+2?",
        ground_truth_answer="4"
    )
    
    # Or use the verify method directly
    is_correct, explanation = verifier.verify(
        node=reasoning_node,
        question="What is 2+2?",
        ground_truth_answer="4"
    )
    ```
"""

from abc import ABC, abstractmethod
from typing import Any

from cot_forge.llm import LLMProvider
from cot_forge.reasoning.types import ReasoningNode


class BaseVerifier(ABC):
  """
  Abstract base class for verifiers that check reasoning correctness.

  This class provides the foundation for implementing different verification
  strategies to evaluate if a reasoning node's answer matches a ground truth.
  All concrete verifiers should inherit from this class and implement the
  verify method.

  Attributes:
      name (str): A unique identifier for the verifier.
      description (str): A human-readable description of the verification method.
      llm_provider (LLMProvider, optional): An LLM provider for verifiers that use 
          language models to check correctness.
      llm_kwargs (dict): Additional parameters to pass to the LLM provider.
  """

  def __init__(self,
               name: str,
               description: str,
               llm_provider: LLMProvider = None,
               llm_kwargs: dict[str, Any] | None = None,
               **kwargs):
    """
    Initialize a verifier with identification and LLM configuration.

    Args:
        name (str): A unique identifier for the verifier.
        description (str): A human-readable description of the verification method.
        llm_provider (LLMProvider, optional): An LLM provider for verifiers that use
            language models to check correctness.
        llm_kwargs (dict, optional): Additional parameters to pass to the LLM provider.
    """
    self.name = name
    self.description = description
    self.llm_provider = llm_provider
    self.llm_kwargs = llm_kwargs or {}

  @abstractmethod
  def verify(self,
             node: ReasoningNode,
             question: str,
             ground_truth_answer: str,
             **kwargs: Any) -> tuple[bool, str]:
    """Verify if the answer is correct."""
    pass

  def __call__(self,
               node: ReasoningNode,
               question: str,
               ground_truth_answer: str,
               **kwargs: Any) -> tuple[bool, str]:
    """Call the verify method."""
    return self.verify(node, question, ground_truth_answer, **kwargs)

  def to_dict(self) -> dict[str, Any]:
    """Convert the verifier to a dictionary representation."""
    return {
        "name": self.name,
        "description": self.description,
        "llm_provider": self.llm_provider.to_dict() if self.llm_provider else None,
        "llm_kwargs": self.llm_kwargs
    }

  @classmethod
  @abstractmethod
  def from_dict(cls, data: dict[str, Any]) -> 'BaseVerifier':
    """Create a verifier instance from a dictionary representation."""
    pass

  def __str__(self) -> str:
    """Return a string representation of the verifier."""
    llm_info = f", llm={self.llm_provider.__class__.__name__}" if self.llm_provider else ""
    return f"{self.name}: {self.description}" + llm_info

  def __repr__(self) -> str:
    """Return a detailed string representation of the verifier."""
    llm_info = f", llm={self.llm_provider.__class__.__name__}" if self.llm_provider else ""
    return f"{self.__class__.__name__}(name='{self.name}', description='{self.description}'{llm_info})"
