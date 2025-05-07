import json
import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

def extract_curly_bracket_content(text: str) -> str:
    """Extracts the curly bracketed json content within a string."""
    # Strip any markdown code block markers
    processed_text = re.sub(r'```json\n|\n```', '', text)
    processed_text = processed_text.strip()
    
    try:
        # Attempt 1: Try to parse as is first
        json.loads(processed_text, strict=False)
        return processed_text
    except json.JSONDecodeError as e_initial:
        # Initial parse failed. Try to repair common LLM mistakes like unescaped newlines.
        # This is a heuristic. Order of replacement can be important for complex cases.
        repaired_text = processed_text.replace('\\', '\\\\') # Must be first: escape literal backslashes
        repaired_text = repaired_text.replace('\r\n', '\\n') # Windows newlines
        repaired_text = repaired_text.replace('\n', '\\n')   # Unix newlines
        repaired_text = repaired_text.replace('\r', '\\r')   # Old Mac newlines / carriage returns
        repaired_text = repaired_text.replace('\t', '\\t')   # Tabs
        # Add other repairs if needed, e.g., for unescaped quotes if that becomes an issue.

        try:
            # Attempt 2: Try to parse the repaired string
            json.loads(repaired_text, strict=False)
            logger.info("Successfully parsed JSON after repairing unescaped control characters.")
            return repaired_text
        except json.JSONDecodeError as e_repaired:
            # If repair also fails, fall back to original behavior (print, log, regex extract)
            logger.warning(
                f"JSON parsing failed for input starting with: '{processed_text[:200]}...'. "
                f"Initial error: {e_initial}. Error after repair: {e_repaired}."
            )
            
            # Original fallback logic:
            pattern = r"\{.*\}" # The original regex
            match = re.search(pattern, processed_text, re.DOTALL)
            if not match:
                # If regex also fails to find a pattern, returning None might be safer
                # than returning a string that's unlikely to be JSON.
                return None 
            json_str = match.group(0)
            # This json_str might still be unparsable by the caller.
            return json_str

def parse_json_response(response: str) -> Any:
    """Extracts json formatting from a reasoning response."""
    try:
        data = json.loads(extract_curly_bracket_content(response), strict=False)
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
    
