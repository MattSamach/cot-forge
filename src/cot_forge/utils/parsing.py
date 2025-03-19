import json
import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

def extract_curly_bracket_content(text: str) -> str:
    """Extracts the curly bracketed json content within a string."""
    pattern = r"\{.*\}"
    match = re.search(pattern, text, re.DOTALL)
    return match.group(0) if match else None

def parse_json_response(response: str) -> Any:
    """Extracts json formatting from a reasoning response."""
    try:
        data = json.loads(extract_curly_bracket_content(response))
        return data
    except (json.JSONDecodeError, TypeError) as err:
        logger.error(f"Error decoding JSON: {err}")
        raise ValueError("Invalid response format. Expected a JSON string.") from err

def extract_cot(response: str) -> dict:
    """
    Extract the chain of thought from a response as 
    a python object (combo of list/dict).
    """
    try:
        data = parse_json_response(response)
        return data.get("CoT", "")
    except AttributeError as err:
        logger.error(f"Attribute error: {err}")
        raise ValueError("Invalid response format. Expected a dictionary with 'CoT' key.") from err
    
def extract_final_answer_from_str(response: str) -> str | None:
    """Extract the final answer from a response."""
    try:
        data = parse_json_response(response)
        
        for action in reversed(data.get("CoT", [])):
            if action.get("action") == "Final Conclusion":
                return action.get("content", "")
            
    except json.JSONDecodeError as err:
        logger.error(f"Error decoding JSON: {err}")
        return None
    
    return None
    
def extract_final_answer_from_cot(data: list[dict]) -> str | None:
    """Extract the final answer from a cot."""
    # Handle error case where data is not a list
    if not isinstance(data, list):
        return None
        
    try:
        for item in reversed(data):
            if item.get("action") == "Final Conclusion":
                return item.get("content", "")
            
    except (KeyError, AttributeError) as err:
        logger.error(f"Error extracting final answer: {err}")
        return None
    
    return None
    
