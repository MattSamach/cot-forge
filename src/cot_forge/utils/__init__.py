"""
Utility functions for parsing and error handling in the CoT Forge project.

This module provides helper functions for:
- Parsing JSON responses and extracting specific content, such as chains of thought (CoT) or final answers.
- Standardized error handling and retry mechanisms for executing operations with fallback strategies.
- Generating and parsing CoT responses from LLM providers.

Modules:
- `parsing.py`: Functions for extracting and parsing JSON content from LLM responses.
- `search_utils.py`: Functions for executing operations with error handling, retries, and fallback mechanisms.

Key Features:
- Extract JSON content from strings and handle malformed responses gracefully.
- Extract specific components, such as CoT or final answers, from parsed JSON data.
- Execute operations with configurable error handling strategies, including retries and fallbacks.
- Generate and parse CoT responses using LLM providers with robust error handling.

These utilities are designed to streamline interactions with LLMs and 
ensure reliability in handling errors and parsing responses.
"""

from .parsing import (
                      extract_cot,
                      extract_final_answer_from_cot,
                      extract_final_answer_from_str,
                      parse_json_response,
)
from .search_utils import execute_with_fallback, generate_and_parse_cot, generate_and_parse_json

__all__ = [
    extract_cot,
    extract_final_answer_from_cot,
    extract_final_answer_from_str,
    parse_json_response,
    execute_with_fallback,
    generate_and_parse_cot,
    generate_and_parse_json
]