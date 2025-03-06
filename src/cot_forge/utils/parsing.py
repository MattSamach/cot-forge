import json
import logging
import re

logger = logging.getLogger(__name__)

def extract_curly_bracket_content(text: str) -> str:
    """Extracts the curly bracketed json content within a string."""
    pattern = r"\{.*\}"
    match = re.search(pattern, text, re.DOTALL)
    return match.group(0) if match else None

def parse_reasoning_response(response: str) -> str:
    """Extracts json formatting from a reasoning response."""
    try:
        data = json.loads(extract_curly_bracket_content(response))
        return data
    except (json.JSONDecodeError, TypeError):
        return ""
    
def extract_cot(response: str) -> str:
    """Extract the chain of thought from a response."""
    try:
        data = parse_reasoning_response(response)
        return data.get("CoT", "")
    except json.JSONDecodeError as err:
        logger.error(f"Error decoding JSON: {err}")
        return None
    
def extract_final_answer_from_str(response: str) -> str:
    """Extract the final answer from a response."""
    try:
        data = parse_reasoning_response(response)
        
        for action in reversed(data.get("CoT", [])):
            if action.get("action") == "Final Conclusion":
                return action.get("content", "")
            
    except json.JSONDecodeError as err:
        logger.error(f"Error decoding JSON: {err}")
        return None
    
def extract_final_answer_from_cot(data: list[dict]) -> str:
    """Extract the final answer from a cot."""
    try:
        for item in reversed(data):
            if item.get("action") == "Final Conclusion":
                return item.get("content", "")
    except KeyError as err:
        logger.error(f"Key error: {err}")
        return None
    
