"""
Rules-based verification implementations for the CoT-Forge reasoning framework.

This module provides verifier implementations that utilize rule-based methods
to assess the correctness of reasoning outputs.
"""

import logging
from typing import Any

from cot_forge.reasoning.types import ReasoningNode
from cot_forge.reasoning.verifiers.base import BaseVerifier
from cot_forge.utils import extract_final_answer_from_cot

logger = logging.getLogger(__name__)


class ExactMatchVerifier(BaseVerifier):
  """
  A verifier that uses an LLM to judge answer correctness and provide feedback.

  This verifier delegates the verification task to an LLM, which compares a final
  answer against a ground truth answer and provides both a binary correctness
  assessment and a detailed explanation.

  Attributes:
      prompt_template (str): Template string for formatting verification prompts
      llm_provider (LLMProvider): Provider for LLM access
      llm_kwargs (dict): Additional parameters for LLM calls
  """

  def __init__(
      self,
      match_case: bool = False
  ):
    """
    Initialize the exact match verifier.

    Args:
        match_case (bool, optional): If True, the verifier will consider case
    """
    name = "exact_match_verifier"
    description = "A verifier that checks if a final answer and ground truth answer are exact matches."
    super().__init__(name=name, description=description)
    self.match_case = match_case

  @classmethod
  def from_dict(cls, data: dict[str, Any]) -> 'BaseVerifier':
    """Create a verifier instance from a dictionary representation."""
    return cls(
        match_case=data.get("match_case", False)
    )

  def verify(self,
             node: ReasoningNode,
             ground_truth_answer: str) -> tuple[bool, str]:
    """
    Verify the correctness of a reasoning node's answer with exact match.

    Args:
        node (ReasoningNode): The reasoning node to verify
        ground_truth_answer (str): The correct answer to compare against

    Returns:
        tuple[bool, str]: A tuple containing:
            - boolean indicating if the answer is correct
            - explanation of the verification result
    """
    if not node.cot:
      logger.error("Node.cot is None")
      return False, "Error: Node.cot is None"

    final_answer = extract_final_answer_from_cot(node.cot)
    if final_answer is None:
      logger.warning("No Final Conclusion found in response")
      node.metadata = {
          **(node.metadata or {}),
          "warning": "missing_final_conclusion"
      }
      return False, "Error: No Final Conclusion found in response"

    final_answer = final_answer.strip().strip('.').strip('!').strip('?')
    ground_truth_answer = ground_truth_answer.strip().strip('.').strip('!').strip('?')
    if not self.match_case:
      final_answer = final_answer.lower()
      ground_truth_answer = ground_truth_answer.lower()

    verification_result = "yes" if final_answer == ground_truth_answer else "no"
    explanation = "Final answer matches ground truth answer." if verification_result == "yes" else (
        "Final answer does not match ground truth answer."
    )

    return verification_result == "yes", explanation
