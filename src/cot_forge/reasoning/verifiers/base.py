"""
Abstract base class for verifiers.
Verifiers are used to check the correctness of a reasoning node's answer
against a ground truth answer. This base class defines the abstract
interface that all verifiers must implement.
"""

from abc import ABC, abstractmethod
from typing import Any

from cot_forge.llm import LLMProvider
from cot_forge.reasoning.types import ReasoningNode


class BaseVerifier(ABC):
    """Abstract base class for verifiers."""
    
    @abstractmethod
    def verify(self,
                node: ReasoningNode,
                question: str,
                ground_truth_answer: str,
                llm_provider: LLMProvider | None,
                llm_kwargs: dict[str, Any] | None,
                **kwargs: Any) -> tuple[bool, str]:
        """Verify if the answer is correct."""
        pass
    
    def __call__(self,
                 node: ReasoningNode,
                 question: str,
                 ground_truth_answer: str,
                 llm_provider: LLMProvider = None,
                 llm_kwargs: dict[str, Any] = None,
                 **kwargs: Any) -> tuple[bool, str]:
        """Call the verify method."""
        return self.verify(node, question, ground_truth_answer, llm_provider, llm_kwargs, **kwargs)