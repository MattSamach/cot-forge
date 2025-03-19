"""
This module defines the verifiers used for verifying if the conclusions of 
chains of thought are correct or not. Verifiers may be LLM-based or 
rules based (not yet implemented).

It exports:
    - `LLMJudgeVerifier`: A class for verifiers that use LLM to judge the quality of outputs.
    - `BaseVerifier`: An abstract base class for all verifiers.
"""

from .base import BaseVerifier
from .llm_verifiers import LLMJudgeVerifier

__all__ = [
    "LLMJudgeVerifier",
    "BaseVerifier"
]